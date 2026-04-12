import os
import cv2
import mmcv
import torch
import json
import mimetypes
import collections
import numpy as np
from filterpy.kalman import KalmanFilter
from skimage.filters import threshold_otsu
from argparse import ArgumentParser
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.visualization import Pose3dLocalVisualizer
from dapa import get_dapa_model
from train_stgcn import SingleFrameSTGCN, get_adjacency_matrix

# 1. 加载四个 reference SMPL pose vector
anchor_dir = './anchor_poses'
anchor_sit    = np.load(os.path.join(anchor_dir, 'anchor_pose_sitting.npy'))
anchor_std    = np.load(os.path.join(anchor_dir, 'anchor_pose_standing.npy'))
anchor_supine = np.load(os.path.join(anchor_dir, 'anchor_pose_supine.npy'))
anchor_crawl  = np.load(os.path.join(anchor_dir, 'anchor_pose_crawling.npy'))
anchors = [anchor_sit, anchor_std, anchor_supine, anchor_crawl]
pose_str = ["Seated","Upright","Supine","Prone","Null"]

# 2. COCO-WholeBody (133) → OpenPose25 映射
COCO2OP25 = {
    0:  0,   # Nose
    1: 16,   # left_eye → LEye (16)
    2: 15,   # right_eye → REye (15)
    3: 18,   # left_ear → LEar (18)
    4: 17,   # right_ear → REar (17)
    5:  5,   # left_shoulder → LShoulder (5)
    6:  2,   # right_shoulder → RShoulder (2)
    7:  6,   # left_elbow → LElbow (6)
    8:  3,   # right_elbow → RElbow (3)
    9:  7,   # left_wrist → LWrist (7)
    10: 4,   # right_wrist → RWrist (4)
    11:12,   # left_hip → LHip (12)
    12: 9,   # right_hip → RHip (9)
    13:13,   # left_knee → LKnee (13)
    14:10,   # right_knee → RKnee (10)
    15:14,   # left_ankle → LAnkle (14)
    16:11,   # right_ankle → RAnkle (11)
    17:19,   # left_big_toe → LBigToe (19)
    18:20,   # left_small_toe → LSmallToe (20)
    19:21,   # left_heel → LHeel (21)
    20:22,   # right_big_toe → RBigToe (22)
    21:23,   # right_small_toe → RSmallToe (23)
    22:24,   # right_heel → RHeel (24)
}

# 3. 从 dataset_info 自动生成 5 个 WholeBody 部分索引
BODY17   = list(range(0, 17))
FOOT6    = list(range(17, 23))
FACE68   = list(range(23, 91))
LHAND21  = list(range(91, 112))
RHAND21  = list(range(112, 133))

dapa_adult_path  = './body_models/dapa_infant.pt'
dapa_child_path  = './body_models/dapa_infant.pt'
smpl_mean_params_path = './body_models/smpl_mean_params.npz'

import torch.nn as nn
from torchvision import transforms, models
import kornia
from kornia.geometry.conversions import rotation_matrix_to_angle_axis

def build_binary_resnet50(num_classes=2):
    """
    基于 torchvision 的 ResNet50 构建一个二分类模型，将最后一层 fc 替换为 (in_feats, num_classes)。
    """
    from torchvision.models import ResNet50_Weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--input-dir', type=str, required=True,
        help='Input directory containing videos'
    )
    parser.add_argument(
        '--output-root', type=str, required=True,
        help='Root directory for outputs (will create b_data/ and p_video/)'
    )
    parser.add_argument(
        '--det-config', required=True,
        help='MMDetection config file for 2D person detector'
    )
    parser.add_argument(
        '--det-checkpoint', required=True,
        help='MMDetection checkpoint file for 2D detector'
    )
    parser.add_argument(
        '--pose3d-config', required=True,
        help='MMPose 3D pose estimator config file'
    )
    parser.add_argument(
        '--pose3d-checkpoint', required=True,
        help='MMPose 3D pose estimator checkpoint file'
    )
    parser.add_argument(
        '--cls-checkpoint', type=str, required=True,
        help='Checkpoint file for the child/adult classifier'
    )
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device used for inference (e.g. "cuda:0" or "cpu")'
    )
    parser.add_argument(
        '--bbox-thr', type=float, default=0.5,
        help='Threshold for detection bbox confidence'
    )
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3,
        help='Keypoint visibility threshold (unused here)'
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Whether to show per-frame visualization in a window'
    )
    parser.add_argument(
        '--num-instances', type=int, default=-1,
        help='Max number of 3D poses to visualize per frame (use -1 for all)'
    )
    parser.add_argument('--sharingan-ckpt', required=True, help='Sharingan gaze 模型权重')
    parser.add_argument(
        '--skeleton-checkpoint', type=str, required=True,
        help='Checkpoint file for the skeleton-based child/adult classifier')
    return parser.parse_args()


def predict_child_adult(model, fullbody_crop, transform, device):
    """
    输入：BGR 图（一个人的全身 ROI），输出：("child"/"adult", p_child, p_adult)
    """
    h, w, _ = fullbody_crop.shape
    if h == 0 or w == 0:
        # 如果裁剪区域无效，则返回成人
        return "adult", 0.0, 1.0

    img_rgb = cv2.cvtColor(fullbody_crop, cv2.COLOR_BGR2RGB)
    x = transform(img_rgb).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)  # (1, 2)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (2,)
    idx = int(probs.argmax())
    return ("child" if idx == 1 else "adult"), float(probs[1]), float(probs[0])

def predict_child_skeleton(kpts3d, model, device):
    """
    kpts3d: numpy array of shape (17,3)
    返回: (p_child, p_adult)
    """
    # transpose → (3,17), expand batch → (1,3,17)
    x = torch.from_numpy(kpts3d.T.astype(np.float32)) \
             .unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)                  # (1,2)
        probs  = torch.softmax(logits, 1)  # (1,2)
    p_child = probs[0,1].item()
    p_adult = probs[0,0].item()
    return p_child, p_adult

# 4. 姿态分类函数
def classify_pose_from_joints(j25, pose_vec=None):
    """输入：j25=(25,3)
       输出：pose_id, pose_label"""
    if np.isnan(j25).any():
        return 4, pose_str[4]
    
    # —— 如果给了 pose_vec，就对四个 anchors 全部做距离比对 ——
    if pose_vec is not None:
        # anchors 中顺序为 [Seated, Upright, Supine, Prone]
        dists = [np.linalg.norm(pose_vec - anc) for anc in anchors]
        pid = int(np.argmin(dists))
        #print("anchors comparison analysis implemented")
        return pid, pose_str[pid]
    
    neck, pelvis = j25[1], j25[8]
    v = neck - pelvis
    ang = np.degrees(np.arccos(np.clip(v[2]/np.linalg.norm(v), -1,1)))
    if ang <= 45:
        hip, knee, ank = j25[9], j25[10], j25[11]
        v1, v2 = knee-hip, ank-knee
        leg_ang = np.degrees(np.arccos(np.clip(
            np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1,1)))
        pid = 0 if leg_ang<120 else 1
    else:
        ear_mid = (j25[17] + j25[18]) / 2.0
        view_v  = j25[0] - ear_mid
        head_ang = np.degrees(np.arccos(np.clip(view_v[2]/np.linalg.norm(view_v), -1,1)))
        pid = 2 if head_ang<90 else 3
    return pid, pose_str[pid]

# ============ Gaze 模型相关 ============
from ultralytics import YOLO
from src.modeling.sharingan import Sharingan
from src.utils.common    import spatial_argmax2d, square_bbox
from boxmot import OCSORT
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.cm as cm

IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD  = [0.28674, 0.27776, 0.27995]
DET_THR  = 0.0   # 不再用 confidence threshold

def load_gaze_models(yolo_ckpt, sharingan_ckpt, device):
    # 虽然不跑 YOLO 检测，但还要构造一个 dummy head_det 用于接口统一
    head_det = YOLO(yolo_ckpt).to(device).eval()
    tracker  = None  # 下面我们用已有的 track_id，不再跑新的跟踪
    # 加载 Sharingan
    sharingan = Sharingan(  # 同你原来的参数
      patch_size=16, token_dim=768, image_size=224,
      gaze_feature_dim=512, encoder_depth=12, encoder_num_heads=12,
      encoder_num_global_tokens=0, encoder_mlp_ratio=4.0,
      encoder_use_qkv_bias=True, encoder_drop_rate=0.0,
      encoder_attn_drop_rate=0.0, encoder_drop_path_rate=0.0,
      decoder_feature_dim=128, decoder_hooks=[2,5,8,11],
      decoder_hidden_dims=[48,96,192,384], decoder_use_bn=True
    )
    ckpt = torch.load(sharingan_ckpt, map_location="cpu")
    sd   = {k.replace("model.",""):v for k,v in ckpt["state_dict"].items()}
    sharingan.load_state_dict(sd, strict=True)
    sharingan.to(device).eval()
    return head_det, tracker, sharingan

def predict_gaze(frame: Image.Image, sharingan, head_det, tracker, device: torch.device):
    img_np = np.array(frame)
    # 1) 检测 heads
    results = head_det(img_np)
    boxes   = results[0].boxes.xyxy.cpu().numpy()   # [N,4]
    confs   = results[0].boxes.conf.cpu().numpy()   # [N]
    dets    = np.concatenate([boxes, confs[:,None]], axis=1) if len(boxes) else np.zeros((0,5))
    # 2) tracking
    tracks = tracker.update(
      np.concatenate([dets, np.zeros((dets.shape[0],1))], axis=1),  # pad cls col
      img_np
    )
    if len(tracks)==0:
        return torch.empty((0,2)), torch.empty((0,3)), torch.empty((0,)), torch.empty((0,4)), torch.empty((0,224,224)), np.array([],int)
    pids       = (tracks[:,4]-1).astype(int)
    head_bboxes= torch.from_numpy(tracks[:,:4]).float()
    tb         = square_bbox(head_bboxes, *img_np.shape[:2][::-1])
    # 3) crop & normalize heads
    heads = []
    for bb in tb:
        crop = frame.crop(bb.numpy().astype(int))
        head = TF.resize(TF.to_tensor(crop),(224,224))
        heads.append(head)
    heads = TF.normalize(torch.stack(heads), mean=IMG_MEAN, std=IMG_STD)
    # 4) full image
    img_t = TF.normalize(TF.resize(TF.to_tensor(frame),(224,224)), mean=IMG_MEAN, std=IMG_STD)
    tb    = tb / torch.tensor([img_np.shape[1],img_np.shape[0]]*2,dtype=torch.float32)
    sample={"image":img_t.unsqueeze(0).to(device),
            "heads":heads.unsqueeze(0).to(device),
            "head_bboxes":tb.unsqueeze(0).to(device)}
    with torch.no_grad():
        gv,ghm,inouts = sharingan(sample)
    ghm       = ghm.squeeze(0).cpu()
    gv        = gv.squeeze(0).cpu()
    gp        = spatial_argmax2d(ghm,normalize=True)
    inouts    = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()
    return gp, gv, inouts, head_bboxes, ghm, pids

def draw_gaze(canvas: np.ndarray,
              head_bboxes: torch.Tensor,
              gaze_points: torch.Tensor,
              gaze_vecs: torch.Tensor,
              inouts: torch.Tensor,
              pids: np.ndarray,
              gaze_heatmaps: torch.Tensor,
              alpha=0.5, io_thr=0.5):
    H,W,_ = canvas.shape
    #（可以简单只画注视点和箭头，省去 heatmap）
    gp_np = gaze_points.numpy()
    hb_np = head_bboxes.numpy()
    if gp_np.size:
        gp_xy = (gp_np * np.array([W,H])) .astype(int)
        for i,(x,y,x2,y2) in enumerate(hb_np.astype(int)):
            if inouts[i]>io_thr:
                px,py = gp_xy[i]
                color = (0,0,255)
                cv2.circle(canvas,(px,py),5,color,-1)
                vec   = gaze_vecs[i].numpy(); vec/=np.linalg.norm(vec)+1e-6
                center = ((x+x2)//2,(y+y2)//2)
                cv2.arrowedLine(canvas, center,
                                (center[0]+int(vec[0]*50),center[1]+int(vec[1]*50)),
                                color,2)
    return canvas

def ray_box_intersection(Cx, Cy, dx, dy, x1, y1, x2, y2):
    """
    Cx,Cy: 射线起点；dx,dy: 单位方向；x1,y1,x2,y2: box 坐标
    返回第一个正向相交点 (Px,Py)
    """
    ts = []
    # left & right
    if dx>0:
        t = (x2 - Cx)/dx
        ts.append(t)
    elif dx<0:
        t = (x1 - Cx)/dx
        ts.append(t)
    # top & bottom
    if dy>0:
        t = (y2 - Cy)/dy
        ts.append(t)
    elif dy<0:
        t = (y1 - Cy)/dy
        ts.append(t)
    # 找最小正 t
    t_pos = [t for t in ts if t>0]
    if not t_pos:
        return Cx, Cy
    t_min = min(t_pos)
    return Cx + dx*t_min, Cy + dy*t_min

def gen_new_color(existing_colors):
    # 简单随机，或者用 HSV 均匀采样
    while True:
        c = tuple(np.random.choice(range(50,256), size=3).tolist())
        if c not in existing_colors:
            return c

import warnings
warnings.filterwarnings("ignore", message=".*dist attribute.*")

def main():
    args = parse_args()

    # ------------------------------------------------
    # 1. 创建输出目录（b_data/ 和 p_video/）
    # ------------------------------------------------
    os.makedirs(args.output_root, exist_ok=True)

    # ------------------------------------------------
    # 2. 初始化 2D 检测器 (MMDetection)
    # ------------------------------------------------
    device = args.device
    from mmpose.utils import adapt_mmdet_pipeline

    detector = init_detector(
        args.det_config, args.det_checkpoint, device=device
    )
    # 把 MMDetection pipeline 转为 MMPose 可识别格式
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # ------------------------------------------------
    # 3. 初始化 3D 姿态估计器 (MMPose)
    # ------------------------------------------------
    pose_estimator = init_model(
        args.pose3d_config, args.pose3d_checkpoint, device=device
    )
    # 开启可视化与 OKS 跟踪
    pose_estimator.cfg.model.test_cfg.mode = 'vis'
    pose_estimator.cfg.model.test_cfg.use_oks_tracking = True
    pose_estimator.cfg.model.test_cfg.tracking_thr = 0.6

    pose_estimator.cfg.visualizer.radius = 3
    pose_estimator.cfg.visualizer.line_width = 2

    det_kpt_color = pose_estimator.dataset_meta.get('keypoint_colors', None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get('skeleton_links', None)
    det_dataset_link_color = pose_estimator.dataset_meta.get('skeleton_link_colors', None)

    pose_estimator.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_estimator.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.det_dataset_link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.skeleton = det_dataset_skeleton
    pose_estimator.cfg.visualizer.link_color = det_dataset_link_color
    pose_estimator.cfg.visualizer.kpt_color = det_kpt_color
    #print(pose_estimator.cfg.test_dataloader.dataset)

    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    
    adult_model = get_dapa_model(dapa_adult_path, smpl_mean_params_path).to(device).eval()
    infant_model = get_dapa_model(dapa_child_path, smpl_mean_params_path).to(device).eval()

    # ------------------------------------------------
    # 4. 初始化 Child/Adult 二分类模型 (ResNet50)
    # ------------------------------------------------
    classifier = build_binary_resnet50(num_classes=2).to(device)
    classifier.load_state_dict(
        torch.load(args.cls_checkpoint, map_location=device)
    )
    classifier.eval()

    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # ─── Skeleton-based 分类模型（Multi-Head GAT + LayerNorm + Residual GCN + EMA） ────
    A = get_adjacency_matrix().to(device)
    skeleton_model = SingleFrameSTGCN(in_c=3, num_class=2, A=A).to(device)
    skeleton_model.load_state_dict(
        torch.load(args.skeleton_checkpoint, map_location=device)
    )
    skeleton_model.eval()
    
    # ------------------------------
    # 初始化 Gaze 模型
    # ------------------------------
    gaze_sharingan = Sharingan(  # 和你原来 load_sharingan_model 一模一样
        patch_size=16, token_dim=768, image_size=224,
        gaze_feature_dim=512, encoder_depth=12, encoder_num_heads=12,
        encoder_num_global_tokens=0, encoder_mlp_ratio=4.0,
        encoder_use_qkv_bias=True, encoder_drop_rate=0.0,
        encoder_attn_drop_rate=0.0, encoder_drop_path_rate=0.0,
        decoder_feature_dim=128, decoder_hooks=[2,5,8,11],
        decoder_hidden_dims=[48,96,192,384], decoder_use_bn=True
    )
    ckpt = torch.load(args.sharingan_ckpt, map_location='cpu')
    sd   = {k.replace('model.',''):v for k,v in ckpt['state_dict'].items()}
    gaze_sharingan.load_state_dict(sd, strict=True)
    gaze_sharingan.to(device).eval()
    face_det = YOLO('yolov11l-face.pt').to(device).eval()
    
    # 全局跟踪与分类缓存
    track_kf = {}        # { tid: KalmanFilter 实例 }
    track_last_seen = {} # { tid: 最近出现的帧号 }
    track_age = {}       # { tid: 已存活帧计数 }
    max_age = 30         # 关键帧重连最大间隔
    score_hist = {}      # { tid: deque([最近若干帧的 child_score], maxlen=10) }
    label_lock = {}      # { tid: "child"/"adult" }
    stable_count = {}   # { tid: 连续复查与当前标签一致的次数 }
    hard_lock = {}      # { tid: 是否已经彻底锁定，不再允许修改 }
    next_tid = 0
    track_colors = {}  # tid -> (B,G,R)
    
    # ------------------------------------------------
    # 5. 遍历输入目录下所有视频
    # ------------------------------------------------
    video_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return

    for video_name in video_files:
        video_path = os.path.join(args.input_dir, video_name)
        video_basename = os.path.splitext(video_name)[0]

        # 创建该视频的输出目录
        out_json_dir = os.path.join(args.output_root, 'b_data', video_basename)
        out_video_dir = os.path.join(args.output_root, 'p_video')
        os.makedirs(out_json_dir, exist_ok=True)
        os.makedirs(out_video_dir, exist_ok=True)

        out_json_path = os.path.join(
            out_json_dir, f'{video_basename}_labels.json'
        )
        out_video_path = os.path.join(
            out_video_dir, f'{video_basename}_vis.mp4'
        )

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width+height, height))

        frame_idx = 0
        
        # 每个视频重新初始化全局跟踪与分类缓存
        track_kf.clear()
        track_last_seen.clear()
        track_age.clear()
        score_hist.clear()
        label_lock.clear()
        stable_count.clear()
        hard_lock.clear()
        next_tid = 0
        
        # 初始化 JSON 结构，按帧追加
        json_dict = {
            "video_name": video_name,
            "fps:": fps,
            "frames": []
        }
        
        while cap.isOpened():
            '''
            # 逐帧处理
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1
            '''
            
            # 每秒一帧处理（每秒取第0帧）
            success, frame = cap.read()
            if not success:
                break
            if frame_idx % int(round(fps)) != 0:
                frame_idx += 1
                continue
            
            if frame_idx % 1000 == 0:
                print(f"Processing frame {frame_idx} of video {video_basename} …")

            # ----------------------------------------
            # 5.1 2D 检测 (inference_detector)
            # ----------------------------------------
            det_result = inference_detector(detector, frame)
            det_instances = det_result.pred_instances.cpu().numpy()

            keep_mask = np.logical_and(
                det_instances.labels == 0,
                det_instances.scores > args.bbox_thr
            )
            raw_bboxes = det_instances.bboxes[keep_mask]
            bboxes = det_instances.bboxes[keep_mask]  # (N,4)

            # 如果这一帧没人，执行关键帧重连检查
            if len(raw_bboxes) == 0:
                # === 关键帧重连：当前帧无检测，更新所有 track_last_seen 但不删除，待后续帧复连 ===
                for tid in list(track_last_seen.keys()):
                    # 如果超过 max_age 帧未见，则彻底删除该 track
                    if frame_idx - track_last_seen[tid] > max_age:
                        track_kf.pop(tid, None)
                        track_last_seen.pop(tid, None)
                        track_age.pop(tid, None)
                        score_hist.pop(tid, None)
                        label_lock.pop(tid, None)
                # 写空帧 JSON 并显示
                json_dict["frames"].append({
                    "frame_index": frame_idx,
                    "persons": []
                })
                writer.write(frame)
                if args.show:
                    cv2.imshow('vis', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue

            # ----------------------------------------
            # 5.1.1 用卡尔曼预测所有活跃 track 到当前帧位置
            # ----------------------------------------
            preds = {}  # {tid: [cx, cy, w, h]} 预测框（中心+宽高）
            for tid, kf in list(track_kf.items()):
                pred = kf.predict()
                cx, cy, w, h = kf.x[0,0], kf.x[1,0], kf.x[2,0], kf.x[3,0]
                preds[tid] = [cx, cy, w, h]

            # ----------------------------------------
            # 5.1.2 匹配：用 IoU 关联 raw_bboxes 和 preds，对应出 track_id
            # ----------------------------------------
            unmatched_raw = set(range(len(raw_bboxes)))
            unmatched_tids = set(preds.keys())
            matches = {}  # { raw_idx: matched_tid }

            # 构造 raw 中点格式以便计算 IoU
            raw_xywh = []
            for box in raw_bboxes:
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                raw_xywh.append([x1 + w / 2, y1 + h / 2, w, h])
            raw_xywh = np.array(raw_xywh)

            # 先计算所有 raw vs preds 的 IoU 矩阵
            iou_mat = np.zeros((len(raw_xywh), len(preds)), dtype=float)
            tids_list = list(preds.keys())
            for i, (cx, cy, w, h) in enumerate(raw_xywh):
                x1_r, y1_r = cx - w/2, cy - h/2
                x2_r, y2_r = cx + w/2, cy + h/2
                for j, tid in enumerate(tids_list):
                    cx_p, cy_p, w_p, h_p = preds[tid]
                    x1_p, y1_p = cx_p - w_p/2, cy_p - h_p/2
                    x2_p, y2_p = cx_p + w_p/2, cy_p + h_p/2
                    # 计算 IoU
                    xx1 = max(x1_r, x1_p)
                    yy1 = max(y1_r, y1_p)
                    xx2 = min(x2_r, x2_p)
                    yy2 = min(y2_r, y2_p)
                    inter_w = max(0, xx2 - xx1)
                    inter_h = max(0, yy2 - yy1)
                    inter = inter_w * inter_h
                    area_r = w * h
                    area_p = w_p * h_p
                    union = area_r + area_p - inter
                    if union > 0:
                        iou_mat[i, j] = inter / union

            # 匹配阈值
            iou_thr = 0.3
            # 贪心选最大 IoU 进行匹配
            for _ in range(min(len(raw_xywh), len(preds))):
                idx_flat = np.argmax(iou_mat)
                i, j = np.unravel_index(idx_flat, iou_mat.shape)
                if iou_mat[i, j] < iou_thr:
                    break
                matched_tid = tids_list[j]
                matches[i] = matched_tid
                unmatched_raw.discard(i)
                unmatched_tids.discard(matched_tid)
                # 置零整行整列以防重复匹配
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1

            # ----------------------------------------
            # 5.1.3 为 unmatched_raw 分配新 ID，并为 unmatched_tids 增加 age
            # ----------------------------------------
            for i in unmatched_raw:
                # 新建卡尔曼对该 raw 进行初始化
                kf = KalmanFilter(dim_x=7, dim_z=4)
                # 状态向量 [cx, cy, w, h, vx, vy, vw]，仅初始化前四维
                kf.x[:4] = np.array(raw_xywh[i]).reshape(4, 1)
                # 状态转移矩阵
                kf.F = np.eye(7)
                kf.F[0, 4] = 1  # cx += vx
                kf.F[1, 5] = 1  # cy += vy
                kf.F[2, 6] = 1  # w  += vw
                # 测量矩阵
                kf.H = np.zeros((4, 7))
                kf.H[0, 0] = 1
                kf.H[1, 1] = 1
                kf.H[2, 2] = 1
                kf.H[3, 3] = 1
                # 预测、测量噪声矩阵，可按需调小
                kf.P *= 10.0
                kf.R *= 5.0
                kf.Q *= 0.01

                new_tid = next_tid
                next_tid += 1
                track_kf[new_tid] = kf
                track_last_seen[new_tid] = frame_idx
                track_age[new_tid] = 0
                matches[i] = new_tid

            # 对 unmatched_tids（上一帧存在，本帧未匹配到）做 age+1，如果超过 max_age 则删除
            for lost_tid in list(unmatched_tids):
                track_age[lost_tid] += 1
                if track_age[lost_tid] > max_age:
                    track_kf.pop(lost_tid, None)
                    track_last_seen.pop(lost_tid, None)
                    track_age.pop(lost_tid, None)
                    score_hist.pop(lost_tid, None)
                    label_lock.pop(lost_tid, None)

            # ----------------------------------------
            # 5.1.4 最后一步，用 matches 决定当前这帧 raw_box 对应的 tid
            #     并用 KalmanFilter 更新 matched 轨道
            # ----------------------------------------
            final_tids = []
            for i, box in enumerate(raw_bboxes):
                cx, cy, w, h = raw_xywh[i]
                assigned_tid = matches[i]
                final_tids.append(assigned_tid)
                # 更新卡尔曼滤波器：把当前测量值 [cx,cy,w,h] 传入 update
                kf = track_kf[assigned_tid]
                kf.update(np.array([cx, cy, w, h]).reshape(4, 1))
                # 重置 age 并记录最后一次出现帧号
                track_age[assigned_tid] = 0
                track_last_seen[assigned_tid] = frame_idx
                
            associated_tids = final_tids  # 当前这 N 个检测框对应的 track_id 列表
            
            # ----------------------------------------
            # 5.2 2D→3D 姿态推理 (inference_topdown)
            #   注：去掉不被支持的 return_heatmap 等参数
            # ----------------------------------------
            xywh_bboxes = []
            for box in bboxes:
                x1, y1, x2, y2 = box.astype(int)
                w, h = x2 - x1, y2 - y1
                xywh_bboxes.append([
                    x1 + w / 2,
                    y1 + h / 2,
                    w,
                    h
                ])
            xywh_bboxes = np.array(xywh_bboxes)
            
            pose2d_results = inference_topdown(
                pose_estimator,
                frame,
                bboxes,
                bbox_format='xyxy'
            )

            # ----------------------------------------
            # 5.3 后处理 3D 关键点：reshape -> 坐标变换 -> rebase -> 扩 batch 维度
            # ----------------------------------------
            for idx, res in enumerate(pose2d_results):
                tid = associated_tids[idx]  # 直接取卡尔曼匹配出来的 id
                res.track_id = tid
                kpts = res.pred_instances.keypoints
                # 统一 reshape 为 (num_joints, 3)
                kpts = kpts.reshape(-1, 3)
                # 转换坐标顺序： (x, y, conf) -> (-x, z, y)
                kpts = -kpts[..., [0, 2, 1]]
                # 让最低点落地 (z 轴 rebase)
                kpts[..., 2] -= np.min(kpts[..., 2], axis=-1, keepdims=True)
                # 扩展为 (1, num_joints, 3)，匹配 InstanceData
                pose2d_results[idx].pred_instances.keypoints = kpts[np.newaxis, ...]

            merged = merge_data_samples(pose2d_results)
            instances_3d = merged.get('pred_instances', None)
            #print(f'Frame {frame_idx}, merged.pred_instances.keypoints.shape =', instances_3d.keypoints.shape)
            #print('→ few example 3D points:\n', instances_3d.keypoints[0, :5])
            # instances_3d.keypoints: (M, num_joints, 3)

            # ----------------------------------------
            # 5.4 Child/Adult 分类 & 标签锁定
            # ----------------------------------------
            tmp_info = []  # [(tid, bbox, kpts3d, label, score_child, score_adult, plabel), ...]

            for idx, res in enumerate(pose2d_results):
                tid = associated_tids[idx]  # 取卡尔曼匹配出来的 id
                res.track_id = tid
                kpts3d = instances_3d.keypoints[idx]  # (J,3)
                
                # ——— 映射到 OpenPose25
                j25 = np.zeros((25,3), dtype=float)
                for coco_i, op_i in COCO2OP25.items():
                    j25[op_i] = kpts3d[coco_i]
                # 补 Neck(1) 和 Pelvis(8)
                j25[1] = (j25[5] + j25[2]) / 2.0
                j25[8] = (j25[12] + j25[9]) / 2.0
                                
                cx, cy, w_box, h_box = xywh_bboxes[idx]
                x = int(cx - w_box / 2)
                y = int(cy - h_box / 2)
                w_box = int(w_box)
                h_box = int(h_box)

                # 1) 计算头身高比
                head_pt       = kpts3d[0]
                shoulder_mid  = (kpts3d[5] + kpts3d[6]) / 2.0
                head_height   = float(np.linalg.norm(head_pt - shoulder_mid))
                hip_mid       = (kpts3d[11] + kpts3d[12]) / 2.0
                body_height   = float(np.linalg.norm(shoulder_mid - hip_mid)) + 1e-6
                head_body_ratio = head_height / body_height
                hb_min, hb_max = 0.15, 0.65
                norm_ratio_hb = np.clip((head_body_ratio - hb_min) / (hb_max - hb_min), 0, 1)
                
                # 2) 计算头肩宽比（耳朵间距 / 肩宽）
                #    COCOWholeBody: 3=左耳, 4=右耳, 5=左肩, 6=右肩
                ear_l, ear_r = kpts3d[3], kpts3d[4]
                head_width   = float(np.linalg.norm(ear_l - ear_r)) + 1e-6
                shoulder_width = float(np.linalg.norm(kpts3d[5] - kpts3d[6])) + 1e-6
                head_shoulder_ratio = head_width / shoulder_width
                hs_min, hs_max = 0.05, 0.75    # 可根据分布微调
                norm_ratio_hw = np.clip((head_shoulder_ratio - hs_min) / (hs_max - hs_min), 0, 1)
                
                '''
                print(f"[DBG_RATIO] Tid={tid}"
                 f"  head_body_ratio={head_body_ratio:.3f} norm_hb={norm_ratio_hb:.3f}"
                      f"  head_shoulder_ratio={head_shoulder_ratio:.3f} norm_hw={norm_ratio_hw:.3f}")
                '''
                      
                # 3) 二分类网络预测
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w_box)
                y2 = min(frame.shape[0], y + h_box)
                fullbody_crop = frame[y1:y2, x1:x2]
                cls_label, p_child, p_adult = predict_child_adult(
                    classifier, fullbody_crop, cls_transform, device
                )
                
                # 先把 fullbody_crop 转成 DAPA 要的输入
                img_rgb = cv2.cvtColor(fullbody_crop, cv2.COLOR_BGR2RGB)
                hps_input = cls_transform(img_rgb).unsqueeze(0).to(device)  # (1,3,224,224)

                # 选用成人或儿童模型（这里直接用 label_lock，也可以用其他逻辑）
                if label_lock.get(tid) == 'child':
                    hps_model = infant_model
                else:
                    hps_model = adult_model
                # DAPA 推理
                with torch.no_grad():
                    pred_rotmat, pred_betas, pred_camera = hps_model(hps_input)
                # pred_rotmat: Tensor of shape (1, 24, 3, 3)
                
                # 把 pred_rotmat reshape 成 (24,3,3)，直接送 kornia 转 axis-angle
                # pred_rotmat: (1, 24, 3, 3) -> view 成 (24,3,3)
                rot_mats = pred_rotmat.view(-1, 3, 3)                      # (24,3,3)
                axis = rotation_matrix_to_angle_axis(rot_mats)             # (24,3)
                axis = axis.contiguous().view(1, -1)                       # (1,72)
                body_pose = axis[0, 3:].cpu().numpy()                      # (69,)

                # 最后，把 SMPL body_pose 作为 pose_vec 传给你的分类函数
                pid, plabel = classify_pose_from_joints(j25, pose_vec=body_pose)

                # 4) 四分量加权融合
                p_child_sk, p_adult_sk = predict_child_skeleton(
                    kpts3d[:17], skeleton_model, device
                )
                
                # 三路融合：head/body, head/shoulder, image, skeleton
                w_hb, w_hw, w_cls, w_sk = 0.03, 0.07, 0.00, 0.90
                score_child = (w_hb * norm_ratio_hb
                             + w_hw * norm_ratio_hw
                             + w_cls * p_child
                             + w_sk  * p_child_sk)
                score_child = np.clip(score_child, 0.0, 1.0)
                score_adult = 1.0 - score_child
                
                # 在 score_child 计算之后立刻加：
                # print(f"[DEBUG] Tid={tid}  raw p_child={p_child:.3f}  raw p_child_sk={p_child_sk:.3f} -> score_child={score_child:.3f}")

                # 把该 tid 的滑动窗口插入本帧 score_child
                if tid not in score_hist:
                    score_hist[tid] = collections.deque(maxlen=100)
                score_hist[tid].append(score_child)
                
                # print(f"[DEBUG] Tid={tid}  score_hist (len={len(score_hist[tid])}): {list(score_hist[tid])}")
                
                # 如果这是一个新出现的 track，要把 stable_count 和 hard_lock 初始化：
                if tid not in stable_count:
                    stable_count[tid] = 0
                if tid not in hard_lock:
                    hard_lock[tid] = False

                # 如果 tid 已经有锁定标签，就带上旧标签；否则先暂不决定（后续锁定逻辑再更新）
                old_label = label_lock.get(tid, None)
                tmp_info.append((tid, (x, y, w_box, h_box), kpts3d, old_label, score_child, score_adult, plabel))
                
            all_scores = np.hstack([list(hist) for hist in score_hist.values()]) if score_hist else np.array([])
            if len(all_scores) >= 10:
                thr = threshold_otsu(all_scores)
            else:
                thr = 0.5  # 数据太少时退回默认
            #print(f"[DEBUG] Frame {frame_idx}: Otsu 动态阈值 = {thr:.3f}")

            # ----------------------------------------
            # 5.4.1 单人场景：先看滑动窗口，如果足够稳才锁定
            # ----------------------------------------
            if len(tmp_info) == 1:
                tid, bbox, k3d, old_label, sc, sa, plabel = tmp_info[0]
                if hard_lock.get(tid, False):
                    # 既然已经硬锁定，就直接使用 label_lock[tid]，不再计算滑窗
                    label = label_lock[tid]
                else:
                    if old_label is None:
                        # 查看滑动窗口里 child_score 的比例
                        hist = list(score_hist.get(tid, []))                        
                        if len(hist) >= 10:
                            child_frac = sum(1 for v in hist if v > thr) / len(hist)
                            #print(f"[DEBUG] Tid={tid}  child_frac={child_frac:.3f}")
                            
                            if child_frac > 0.5:
                                label_lock[tid] = 'child'
                            elif (1 - child_frac) > 0.5:
                                label_lock[tid] = 'adult'
                            else:
                                label_lock[tid] = None  # 先不决定
                        else:
                            label_lock[tid] = None    # 数据不足，先不决定
                        label = label_lock[tid]
                    else:
                        label = old_label
                tmp_info[0] = (tid, bbox, k3d, label, sc, sa, plabel)

            # ----------------------------------------
            # 5.4.2 多人场景：优先保留已有 child，其次滑窗 + 差距筛选锁定
            # ----------------------------------------
            else:
                num = len(tmp_info)
                # 先收集已经锁定的 child / adult
                locked_children = []
                locked_adults = []
                for i, (tid, bbox, k3d, old_label, sc, sa, plabel) in enumerate(tmp_info):
                    if label_lock.get(tid) == 'child':
                        locked_children.append(i)
                    elif label_lock.get(tid) == 'adult':
                        locked_adults.append(i)

                if locked_children:
                    # 在所有已锁 child 里，选一个最可能的
                    best_score = -1
                    best_idx   = locked_children[0]
                    for i in locked_children:
                        tid, *_ , sc, _ , _ = tmp_info[i]
                        hist = list(score_hist.get(tid, []))
                        if len(hist) >= 10:
                            child_frac = sum(1 for v in hist if v > thr) / len(hist)
                        else:
                            child_frac = sc
                        # 先比滑窗 child_frac，再比当帧 sc
                        if (child_frac, sc) > (best_score, tmp_info[best_idx][4]):
                            best_score = child_frac
                            best_idx   = i
                else:
                    # 没有锁成 child，但若有锁成 adult 的，就剔除这些 adult
                    if locked_adults:
                        candidates = [i for i in range(num) if i not in locked_adults]
                    else:
                        # 完全没锁过，所有人都是候选
                        candidates = list(range(num))
                        
                    if not candidates:
                        candidates = list(locked_adults)
                        
                    # 从 candidates 中选一个最可能是 child 的
                    best_score = -1
                    # print(f"[DEBUG] frame {frame_idx}: candidates = {candidates}")
                    best_idx = candidates[0]
                    for i in candidates:
                        tid, bbox, k3d, old_label, sc, sa, plabel = tmp_info[i]
                        hist = list(score_hist.get(tid, []))
                        if len(hist) >= 10:
                            child_frac = sum(1 for v in hist if v > thr) / len(hist)
                        else:
                            child_frac = sc
                        # 先比滑窗 child_frac，再比当帧 sc
                        if (child_frac, sc) > (best_score, tmp_info[best_idx][4]):
                            best_score = child_frac
                            best_idx = i

                # 最终锁定：best_idx 为 child，其它都是 adult
                for i, (tid, bbox, k3d, old_label, sc, sa, plabel) in enumerate(tmp_info):
                    if hard_lock.get(tid, False):
                        # 硬锁后绝不改变
                        label_here = label_lock[tid]
                    else:
                        new_label = 'child' if i == best_idx else 'adult'
                        # 只有真正切换时才清零 stable_count
                        if label_lock.get(tid) != new_label:
                            stable_count[tid] = 0
                        label_lock[tid] = new_label
                        label_here = new_label

                    tmp_info[i] = (tid, bbox, k3d, label_here, sc, sa, plabel)

            # ----------------------------------------
            # 5.5 写入当帧 JSON
            # ----------------------------------------
            persons_this_frame = []
            for tid, (x, y, w_box, h_box), k3d, label, sc, sa, plabel in tmp_info:
                persons_this_frame.append({
                    "track_id": int(tid),
                    "label": label,
                    "pose_label": plabel,
                    "bbox": [int(x), int(y), int(w_box), int(h_box)],
                    "keypoints_3d": k3d.tolist()
                })
            json_dict["frames"].append({
                "frame_index": frame_idx,
                "persons": persons_this_frame
            })

            
            # -------------------------------
            # 5.6 用 visualizer 画 2D/3D 关键点
            # -------------------------------
            visual_frame_rgb = mmcv.bgr2rgb(frame.copy())  # 转成 RGB
            #print(f'[Debug] Frame {frame_idx}: visual_frame_rgb.shape = {visual_frame_rgb.shape}')
        
            visualizer.add_datasample(
                name='result',                           # 任意不重复的键
                image=visual_frame_rgb,                  # 传原图 RGB，作为 2D 底图，同时 _draw_3d_data_samples 也会拿它的高度 H 来生成 3D 画布
                data_sample=merged,                      # merged 包含 pred_instances.keypoints(3D)，供 3D 绘制
                det_data_sample=merged,                  # merged 也包含 pred_instances.keypoints(2D)，供 2D 绘制
                draw_gt=False,                           # 不画 GT
                draw_pred=True,                          # 画 3D Prediction
                draw_2d=True,                            # 画 2D
                draw_bbox=False,                          # 2D 时画 Detect BBox
                show_kpt_idx=False,
                dataset_2d=pose_estimator.dataset_meta['dataset_name'],  # 2D 模型所用数据集名（如 'topdown_coco_wholebody'）
                dataset_3d=pose_estimator.dataset_meta['dataset_name'],  # 3D 模型所用数据集名（如 'h36m'）
                convert_keypoint=False,                    # 若需要把 2D layout→3D layout 则为 True，否则 False
                axis_azimuth=70,
                axis_limit=400,
                axis_dist=10.0,
                axis_elev=15.0,
                num_instances=-1,                          # 强制把所有人的 3D 都画到一个 H×H 子图里
                show=False,
                wait_time=0,
                out_file=None,
                kpt_thr=args.kpt_thr,
                step=0
            )
            vis_combined_rgb = visualizer.get_image()
            vis_combined_bgr = mmcv.rgb2bgr(vis_combined_rgb)
            #print('>> vis_combined_rgb.shape =', vis_combined_rgb.shape)
            #print(f'[Debug] Frame {frame_idx}: vis_combined_bgr.shape = {vis_combined_bgr.shape}')
            
            # ----------------------------------------
            # 5.7 可视化：在图像上绘制 bbox、track_id、child/adult 标签，以及 3D 投影
            # ----------------------------------------
            #vis_frame = frame.copy()
            
            # ----------------------------------------
            # 5.8 Gaze 推理：先重排序 bboxes & head crops，保持和 tmp_info 同步
            # ----------------------------------------
            H, W = frame.shape[:2]
            
            # ——— 5.8.0 提取人脸框 ———
            face_bboxes = []
            face_found  = []
            # 按 tmp_info 顺序取出对应的 raw_bbox
            ordered_raw = [ raw_bboxes[i] for i in range(len(tmp_info)) ]
            
            for box in ordered_raw:
                x1,y1,x2,y2 = box.astype(int)
                crop = frame[y1:y2, x1:x2]                # 取 person ROI
                # 在 ROI 内检测 face
                results = face_det(crop, conf=0.3, iou=0.45, verbose=False)
                boxes_f  = results[0].boxes.xyxy.cpu().numpy()
                confs_f  = results[0].boxes.conf.cpu().numpy()
                if len(boxes_f):
                    # 选置信度最高的那个 face
                    idx = int(confs_f.argmax())
                    fx1,fy1,fx2,fy2 = boxes_f[idx]
                    # ROI→全图坐标
                    gx1, gy1 = int(x1 + fx1), int(y1 + fy1)
                    gx2, gy2 = int(x1 + fx2), int(y1 + fy2)
                    face_found.append(True)
                else:
                    # 回退到全身框中心区域
                    gx1, gy1, gx2, gy2 = x1, y1, x2, y2
                    face_found.append(False)
                face_bboxes.append([gx1, gy1, gx2, gy2])

            head_bboxes = torch.from_numpy(np.array(face_bboxes)).float().to(device)
            sq_boxes    = square_bbox(head_bboxes, W, H)
            frame_pil   = Image.fromarray(frame[..., ::-1])

            # 裁剪 & normalize
            heads = []
            for bb in sq_boxes:
                x1,y1,x2,y2 = bb.int().tolist()
                crop = frame_pil.crop((x1,y1,x2,y2))
                heads.append(TF.resize(TF.to_tensor(crop),(224,224)))
            if heads:
                heads = TF.normalize(torch.stack(heads).to(device), mean=IMG_MEAN, std=IMG_STD)
            else:
                heads = torch.empty((0,3,224,224), device=device)

            # 整图
            img_t = TF.normalize(
                TF.resize(TF.to_tensor(frame_pil),(224,224)),
                mean=IMG_MEAN, std=IMG_STD
            ).unsqueeze(0).to(device)

            # 归一化 bboxes
            scale = torch.tensor([W,H,W,H], device=device)
            t_bbs  = (sq_boxes/scale).unsqueeze(0).to(device)

            # 一次性推理 N 个人
            with torch.no_grad():
                gv, ghm, inouts = gaze_sharingan({
                    "image":       img_t,
                    "heads":       heads.unsqueeze(0),
                    "head_bboxes": t_bbs
                })
            gv, ghm = gv.squeeze(0).cpu(), ghm.squeeze(0).cpu()
            inouts = torch.sigmoid(inouts.squeeze(0)).cpu()
            gp     = spatial_argmax2d(ghm, normalize=True)  # (N,2)

            # 再画一遍所有 track
            for i, (tid, (x,y,w_box,h_box), k3d, label, sc, sa, plabel) in enumerate(tmp_info):

                # —— 1) 根据 child/adult 分配颜色 —— 
                # 粉紫色 (magenta) 给成人，青色 (cyan) 给儿童
                if label == 'adult':
                    col = (255, 0, 255)   # BGR 中的 magenta
                else:
                    col = (255, 255, 0)   # BGR 中的 cyan

                # —— 2) 画 Face bbox（用同样的 col）——
                fx1,fy1,fx2,fy2 = face_bboxes[i]
                cv2.rectangle(vis_combined_bgr,
                              (fx1,fy1),(fx2,fy2),
                              col, 2)
                    
                # —— 3) 画 bbox + ID/label —— 
                cv2.rectangle(vis_combined_bgr,(x,y),(x+w_box,y+h_box),col,2)
                cv2.putText(vis_combined_bgr,
                            f"ID:{tid}-{label}-{plabel}",
                            (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col,2)
                
                # —— 写入 face_bbox 到 JSON —— 
                w_f, h_f = fx2 - fx1, fy2 - fy1
                json_dict["frames"][-1]["persons"][i]["face_bbox"] = [int(fx1), int(fy1), int(w_f), int(h_f)]

                # —— 4) 画 Gaze 点 & 箭头 —— 
                if face_found[i] and inouts[i] > 0.5:  # 只有检测到 face 且置信度足够才画 arrow
                    # 注视点
                    px,py = (gp[i].numpy() * np.array([W,H])).astype(int)
                    cv2.circle(vis_combined_bgr, (px,py), 8, col, -1)

                    # 方向向量、face 中心
                    vx, vy = gv[i].numpy()
                    vnorm = np.linalg.norm([vx,vy]) + 1e-6
                    dx, dy = vx/vnorm, vy/vnorm

                    fx1,fy1,fx2,fy2 = face_bboxes[i]
                    Cx = fx1 + (fx2-fx1)/2
                    Cy = fy1 + (fy2-fy1)/2

                    # 算出从 C 沿 (dx,dy) 与 face box 边的交点 P
                    Px, Py = ray_box_intersection(Cx, Cy, dx, dy, fx1, fy1, fx2, fy2)
                    # 再往外画一小段箭头
                    end_x = int(Px + dx * 70)
                    end_y = int(Py + dy * 70)

                    cv2.arrowedLine(vis_combined_bgr,
                                    (int(Px), int(Py)),
                                    (end_x, end_y),
                                    col, 3)
                    # 写回 JSON
                    json_dict["frames"][-1]["persons"][i]["gaze_point"] = [int(px), int(py)]
                    
                # —— 5) 简易 3D 投影 —— 
                for kpt in k3d:
                    px3 = int(x + w_box/2 + kpt[0]*(w_box/2))
                    py3 = int(y + h_box   + kpt[2]*50)
                    cv2.circle(vis_combined_bgr,(px3,py3),2,col,-1)

            # 最终把合成图写入视频
            writer.write(vis_combined_bgr)

            if args.show:
                cv2.imshow('vis', vis_combined_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # === 每 1500 帧做一次“滑窗复查” === （每帧 → 100 每秒 → 1500）
            if frame_idx % 1500 == 0:
                # 1) 收集所有 track 的滑窗 child 分数
                all_scores_t = np.hstack([list(hist) for hist in score_hist.values()]) if score_hist else np.array([])
                if len(all_scores_t) >= 25:  #（每帧 → 50 每秒 → 25）
                    thr_t = threshold_otsu(all_scores_t)
                else:
                    thr_t = 0.5  # 数据太少时退回默认
                # print(f"[DEBUG] Frame {frame_idx}: Otsu 复查动态阈值 = {thr_t:.3f}")
                
                for tid in list(label_lock.keys()):
                    # 对于已经“彻底锁定”的直接跳过
                    if hard_lock.get(tid, False):
                        #print(f"  [Tid {tid}] hard-locked as {label_lock[tid]}, skip recheck")
                        continue

                    hist = list(score_hist.get(tid, []))
                    if len(hist) < 5:
                        # 数据不足，先跳过
                        #print(f"  [Tid {tid}] hist_length={len(hist)} <5, skip recheck")
                        continue

                    child_frac = sum(1 for v in hist if v > thr_t) / len(hist)

                    # 计算“建议标签”
                    if child_frac > 0.5:
                        suggested = 'child'
                    elif (1 - child_frac) > 0.5:
                        suggested = 'adult'
                    else:
                        suggested = None  # 保持不变

                    current = label_lock.get(tid, None)
                    
                    # print(f"  [Tid {tid}] current_label={current}, child_frac={child_frac:.2f}, suggested={suggested}")

                    if suggested is None or suggested == current:
                        # 如果建议标签与当前相同（或无法判断），就把 stable_count+1
                        stable_count[tid] = stable_count.get(tid, 0) + 1
                        # print(f"    -> stable_count[{tid}] becomes {stable_count[tid]}")
                    else:
                        # 如果建议标签与当前不一致，就立即切换，并把计数归零
                        # print(f"    -> switch label of Tid {tid} from {current} to {suggested
                        label_lock[tid] = suggested
                        stable_count[tid] = 0

                    # 若连续三次都没改变，就彻底锁定，不再让它改
                    if stable_count[tid] >= 3 and current is not None:
                        hard_lock[tid] = True
                        # print(f"    -> Tid {tid} hard-locked as {label_lock[tid]}")
                        
            frame_idx += 1
        cap.release()
        writer.release()

        # ----------------------------------------
        # 5.9 写 JSON 文件
        # ----------------------------------------
        with open(out_json_path, 'w') as f:
            json.dump(json_dict, f, indent=2)

        print(f"Saved JSON to {out_json_path}")
        print(f"Saved visualization video to {out_video_path}")

    print("All videos processed.")

if __name__ == '__main__':
    main()
