import os

import cv2
import numpy as np
import PIL.Image as pil_img
import smplx

# from utils.opendr_renderer import SMPLRenderer
import constants as cfg
from hps.body_model import init_body_model

# opendr_renderer = SMPLRenderer(face_path=cfg.smpl_faces)
crop_size = 224  # crop size used in DAPA
infant_bm = init_body_model(model_path=cfg.smil_model_path, batch_size=1, create_body_pose=False).cuda()
adult_bm = init_body_model(model_path=cfg.smpl_model_path, batch_size=1, create_body_pose=False).cuda()


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


def get_result(pred_pose, pred_betas, pred_cam, body_type, x, y, h, target_focal, orig_img_width, orig_img_height):
    """From the model's direct output, get the vertices (3D), joints (3D). Only supports single person.
    """
    pred_pose = pred_pose.unsqueeze(0)
    pred_betas = pred_betas.unsqueeze(0) 
    pred_cam = pred_cam.unsqueeze(0).clone()
    if body_type == 'adult':
        bm = adult_bm(global_orient=pred_pose[:, :3].float(), body_pose=pred_pose[:, 3:].float(), betas=pred_betas)
    elif body_type == 'infant':
        bm = infant_bm(global_orient=pred_pose[:, :3].float(), body_pose=pred_pose[:, 3:].float(), betas=pred_betas)
    else:
        raise Exception('Body type {} not recognized.'.format(body_type))

    joints = bm.joints.detach()[0].detach().cpu().numpy()
    verts = bm.vertices.detach()[0].detach().cpu().numpy()

    trans = get_original(pred_cam[0].detach().cpu().numpy(), x.item(), y.item(), h.item(), target_focal, orig_img_width, orig_img_height)

    return verts, joints, trans


def render_image_dapa(verts, img_path, out_image_path, target_focal, orig_img_width, orig_img_height, save=True):
    framename = img_path.split('/')[-1]
    out_img_fn = os.path.join(out_image_path, 'rendered_{}'.format(framename))
    if os.path.exists(out_img_fn):
        # Another person in this image has been rendered in a previous batch. Render on top of that.
        img = cv2.imread(out_img_fn)
    else:
        img = cv2.imread(img_path)
    img = img[:, :, ::-1]
    cam_for_render = np.hstack([target_focal, np.array([orig_img_width/2, orig_img_height/2])])
    rend_img = opendr_renderer(verts, cam_for_render, img=img, color_id=None)  # (1080, 1920, 3)
    pimg = pil_img.fromarray(rend_img)
    if save:
        print('rendered to ', out_img_fn)
        pimg.save(out_img_fn)
    return pimg


def collect_results_for_image_dapa(pred_pose, pred_betas, pred_cam, pred_transl, batch, target_focal, orig_img_width, orig_img_height):
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
                pred_pose[i], pred_betas[i], pred_cam[i], body_types[i], yhxw[i,2], yhxw[i,0], yhxw[i,1], target_focal,
                orig_img_width, orig_img_height)
            if pred_transl is not None:
                trans = pred_transl[i].detach().cpu().numpy()
            verts_results[img_name].append(verts)
            assert results_batch[i] is None
            results_batch[i] = {
                'betas': pred_betas[i].unsqueeze(0).cpu().numpy(), 
                'pred_cameras': pred_cam[i].unsqueeze(0).cpu().numpy(), 
                'body_pose': pred_pose[i, 3:].unsqueeze(0).cpu().numpy(), 
                'global_orient': pred_pose[i, :3].unsqueeze(0).cpu().numpy(),
                'transl': np.expand_dims(trans, axis=0),
                'joints': np.expand_dims(joints, axis=0),
                'body_type': body_types[i],
                'model_type': 'smil' if body_types[i] == 'infant' else 'smpl',
                'keypoints': keypoints[i].unsqueeze(0).cpu().numpy(),
            }
        
    return verts_results, results_batch