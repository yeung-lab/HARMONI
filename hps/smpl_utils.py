import os

import cv2
import numpy as np
import PIL.Image as pil_img
import smplx
import torch

import constants as cfg
from hps.body_model import init_body_model
from hps.smpla import prepare_smpla_model

crop_size = 224  # crop size used in DAPA
infant_bm = init_body_model(model_path=cfg.smil_model_path, batch_size=1, create_body_pose=False).cuda()
adult_bm = init_body_model(model_path=cfg.smpl_model_path, batch_size=1, create_body_pose=False).cuda()
smpla_bm = prepare_smpla_model(dtype=torch.float32, gender='neutral').cuda()


def get_original(cam, x, y, h, target_focal, orig_img_width, orig_img_height):
    """
    Goal: we want the multiple humans in the image have the same camera specs (f, cx, cy), 
    where cx = 1920/2, cy = 1080/2., f = 1060.
    To do that, we need to 
    (1) first, scale the person in the z-axis to get to the targeted focal length
    (2) then, shift the person in the x, y axis to get to the targeted principal point

    Args:
        cam: direct cam parameters predicted the model

    Outputs
    --------
        trans: translation being applied to this person so that everyone in the frame
                are placed correctly in the same 3D coordinate.
    """
    scale = crop_size / h
    undo_scale = 1. / scale

    flength = 500.
    curr_focal = flength * undo_scale # predicted focal length
    tz = flength / (0.5 * crop_size * cam[0])
    
    trans = np.hstack([cam[1:], tz])
    
    dx = (orig_img_width/2 - (x+h/2)); dy = orig_img_height/2 - (y+h/2)
    trans[2] /= (curr_focal/target_focal)  # first, scale in the axis
    new_tz = tz / (curr_focal/target_focal)
    trans[0] -= new_tz * dx / target_focal  # then, shift
    trans[1] -= new_tz * dy / target_focal  
    return trans


def get_result(pred_pose, pred_betas, pred_cam, body_type, yhxw, target_focal, orig_img_width, orig_img_height, smpl_type, kid_age=1.0):
    """From the model's direct output, get the vertices (3D), joints (3D). Only supports single person.
    """
    pred_pose = pred_pose.unsqueeze(0)
    pred_betas = pred_betas.unsqueeze(0) 
    pred_cam = pred_cam.unsqueeze(0).clone()
    if smpl_type == 'smpl_smil':
        if body_type == 'adult':
            bm = adult_bm(global_orient=pred_pose[:, :3].float(), body_pose=pred_pose[:, 3:].float(), betas=pred_betas)
        elif body_type == 'infant':
            bm = infant_bm(global_orient=pred_pose[:, :3].float(), body_pose=pred_pose[:, 3:].float(), betas=pred_betas)
        else:
            raise Exception('Body type {} not recognized.'.format(body_type))
        joints = bm.joints.detach()[0].detach().cpu().numpy()
        verts = bm.vertices.detach()[0].detach().cpu().numpy()
    else:
        pred_betas = torch.cat([pred_betas, torch.zeros_like(pred_betas[:, :1])], dim=1)
        if body_type == 'infant':
            pred_betas[:, -1] = kid_age
        verts, joints = smpla_bm(pred_betas.float(), pred_pose.float())
        if body_type == 'infant':
            verts[:, :, 1] += 0.2  #  -0.05 -> 1.5
            verts[:, :, 2] += 2.
            
        verts = verts[0].detach().cpu().numpy()
        joints = joints[0].detach().cpu().numpy()

    # change to full image's coordinate
    device = pred_cam.device
    yhxw = torch.from_numpy(yhxw).unsqueeze(0).to(device)
    scale = torch.max(yhxw[:, 1], yhxw[:, 3]) / 200
    cx, cy = yhxw[:, 2] + yhxw[:, 3] / 2., yhxw[:, 0] + yhxw[:, 1] / 2.
    center = torch.stack((cx, cy), dim=-1)
    focal_length = torch.tensor([target_focal], dtype=torch.float32, device=device)#.expand(cur_batch_size)
    full_image_shape = torch.tensor([orig_img_height, orig_img_width]).to(device).float().unsqueeze(0)#.expand(cur_batch_size, -1)
    trans = cam_crop2full(pred_cam, center, scale, full_image_shape, focal_length)

    trans = trans.squeeze(0).cpu().numpy()

    if body_type == 'infant':
        trans[0] *= 0.5  # x axis.
        trans[1] *= 0.3  # y axis.  # 0.2: the child is too high. 0.5. the child is too low.
        trans[2] *= 0.2
    return verts, joints, trans


def collect_results_for_image_dapa(pred_pose, pred_betas, pred_cam, pred_transl, batch, target_focal, orig_img_width, orig_img_height, 
                                    smpl_type='smpla', kid_age=1.0):
    assert smpl_type in ['smpla', 'smpl_smil']
    pred_pose = pred_pose.detach()
    pred_betas = pred_betas.detach()
    if pred_cam is not None:
        pred_cam = pred_cam.detach()
    img_names = np.sort(np.unique(batch['img_name']))
    verts_results = dict()
    yhxw = batch['yhxw'].cpu().numpy()
    # y, h, x, w = batch['yhxw'].cpu().numpy()[0]
    body_types = batch['body_type']
    keypoints = batch['keypoints']
    results_batch = [None for _ in range(len(batch['img_name']))]
    for img_name in img_names:
        idxs = np.where(np.array(batch['img_name'])==img_name)[0]
        verts_results[img_name] = []
        for i in idxs:
            verts, joints, trans = get_result(
                pred_pose[i], pred_betas[i], pred_cam[i], body_types[i], yhxw[i], target_focal,
                orig_img_width, orig_img_height, smpl_type, kid_age)
            if pred_transl is not None:
                trans = pred_transl[i].detach().cpu().numpy()
            verts_results[img_name].append(verts)
            assert results_batch[i] is None
            results_batch[i] = {
                'img_name': img_name,
                'betas': pred_betas[i].unsqueeze(0).cpu().numpy(), 
                'pred_cameras': pred_cam[i].unsqueeze(0).cpu().numpy(), 
                'body_pose': pred_pose[i, 3:].unsqueeze(0).cpu().numpy(), 
                'global_orient': pred_pose[i, :3].unsqueeze(0).cpu().numpy(),
                'transl': np.expand_dims(trans, axis=0),
                'joints': np.expand_dims(joints, axis=0) + np.expand_dims(trans, axis=0),
                'body_type': body_types[i],
                'model_type': 'smil' if body_types[i] == 'infant' else 'smpl',
                'keypoints': keypoints[i].unsqueeze(0).cpu().numpy(),
            }
        
    return verts_results, results_batch



def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N, 1) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam