from collections import defaultdict
import os
import sys
from typing import Tuple

import cv2
from loguru import logger
import numpy as np
from tqdm import tqdm
import torch
import trimesh
import pyrender

from hps.body_model import init_body_model
from hps.smpla import prepare_smpla_model
from utils.geometry import batch_euler2matrix
from visualization.utils import get_colors, get_checkerboard_plane, look_at_camera, rotation_matrix_between_vectors, perspective_projection, rotate_view_weak_perspective
from visualization.keypoints import draw_skeleton
from postprocess.filter_frames import keep_frame
from downstream.calc_downstream import pose_str_map, touch_str_map, visibility_str_map

colors = get_colors()

def render(dataset, results, labels, img_path, save_folder, cfg, cam_params, skip_if_no_infant=False, device='cuda', save_mesh=False,
           camera_center=np.array([960, 540]), img_list=None, fast_render=True, use_smoothed=False,
           add_ground_plane=False, anchor=None, ground_normal=None, top_view=False,
           keep_criterion='all', renderer='pyrender', smpl_type='smpl_smil', kid_age=1.0):
    """Generate meshes from the saved estimated SMPL parameters, and then render.

    anchor: a point on the ground plane
    ground_normal: the normal of the ground plane
    """
    adult_bm = init_body_model(cfg.smpl_model_path, batch_size=1, create_body_pose=True).to(device)
    infant_bm = init_body_model(cfg.smil_model_path, batch_size=1, create_body_pose=True).to(device)
    smpla_bm = prepare_smpla_model(torch.float32, 'neutral').to(device)
    smpl_faces = adult_bm.faces

    img_to_persons = defaultdict(list)
    for k, frame_name in dataset.person_to_img.items():
        img_to_persons[frame_name[0]].append(k)

    img_names = sorted(list(img_to_persons.keys()))
    to_tensor = lambda arr: torch.from_numpy(arr).to(device)
    to_numpy = lambda ts: ts.detach().cpu().numpy()

    if img_list is not None:
        img_names = img_list  # only render images in the given list
    
    for img_name in tqdm(img_names, desc='Rendering'):
        cam_vfov, cam_pitch, cam_roll, cam_focal_length = cam_params
        # cam_vfov, cam_pitch, cam_roll, cam_focal_length = 0.4688836932182312, 0.21909338235855103, -0.005275309085845947, 1060.
        cam_params = np.array([cam_vfov, cam_pitch, cam_roll, cam_focal_length])
        focal_lengths = [cam_focal_length, cam_focal_length]
        render_rotmat = batch_euler2matrix(torch.tensor([[-cam_pitch, 0., cam_roll]]))[0]

        persons = img_to_persons[img_name]
        if skip_if_no_infant and len(persons) == 1: continue

        if labels is not None:
            label_for_frame = labels[img_name]
            if np.isnan(label_for_frame['distance']):
                desc = ['Low confidence frame.']
            else:
                desc = [
                    'Distance between caregiver and child: {:.2f}m'.format(label_for_frame['distance']),
                    'Child Pose: {}'.format(pose_str_map[label_for_frame['pose']]),
                    'Touch: {}'.format(touch_str_map[label_for_frame['touch']]),
                    'Visibility: {}'.format(visibility_str_map[label_for_frame['visibility']]),
                    'Adult child angle: {:.2f} deg, {:.2f} deg'.format(label_for_frame['adult_angle'], label_for_frame['infant_angle'])

                ]
        else:
            desc = ['']

        vertices = []
        faces = []
        mesh_colors = []
        keypoints_2d = []
        camera_translations = []
        track_ids = []
        body_types_for_this_image = [results.results[person_id]['model_type'] for person_id in persons if person_id in results.results]
        if not keep_frame(body_types_for_this_image, keep_criterion):
            continue
        
        for person_id in persons:
            if person_id not in results.results: continue
            
            if use_smoothed:
                result = results.smoothed_results[person_id]
            else:
                result = results.results[person_id]
            is_ghost = dataset[person_id]['is_ghost']
            kp_2d = result['keypoints'][0,:25]
            pose, betas = to_tensor(result['body_pose']), to_tensor(result['betas'])
            global_orient = to_tensor(result['global_orient'])
            faces.append(smpl_faces)

            if smpl_type == 'smpl_smil':
                if result['model_type'] == 'smpl':
                    bm = adult_bm(global_orient=global_orient, body_pose=pose, betas=betas)
                    mesh_color = 'green'
                elif result['model_type'] == 'smil':
                    bm = infant_bm(global_orient=global_orient, body_pose=pose, betas=betas)
                    mesh_color = 'blue'
                else:
                    raise Exception()
                verts = bm.vertices
            else:
                
                if result['model_type'] == 'smil':
                    betas = torch.cat([betas*0, torch.zeros_like(betas[:, :1]) + kid_age], dim=1)
                    full_pose = torch.cat([global_orient, pose], dim=1)
                    verts, _ = smpla_bm(betas, full_pose, root_align=True)
                    verts[:, :, 1] += 0.2  #  -0.05 -> 1.5
                    verts[:, :, 2] += 2.
                    mesh_color = 'purple'
                else:
                    bm = adult_bm(global_orient=global_orient, body_pose=pose, betas=betas)
                    verts = bm.vertices
                    mesh_color = 'green'

            if kp_2d is None or is_ghost:
                mesh_color = 'red'
            elif kp_2d[:,2].sum() > 9: 
                mesh_color = 'dark_'+mesh_color
            elif kp_2d[:,2].sum() < 5:
                mesh_color = 'light_'+mesh_color
            keypoints_2d.append(kp_2d)
            mesh_colors.append(mesh_color)
            camera_translations.append(result['transl'][0])
            vertices.append(to_numpy(verts)[0])
            track_ids.append(dataset.id_to_track[person_id])
        
        save_filename = os.path.join(save_folder, img_name)
        if save_mesh:
            save_mesh_folder = os.path.join(save_folder, 'meshes')
            if not os.path.exists(save_mesh_folder):
                os.makedirs(save_mesh_folder)

            mesh_filename = os.path.join(save_mesh_folder, os.path.splitext(img_name)[0]+f'_{person_id}.obj')
        else:
            mesh_filename = None
        
        render_image_group(
            os.path.join(img_path, img_name), camera_translations, vertices, render_rotmat.numpy(), focal_lengths, camera_center,
            faces=faces, keypoints_2d=keypoints_2d, track_ids=track_ids, save_filename=save_filename,
            mesh_color=mesh_colors, desc=desc, mesh_filename=mesh_filename, fast_render=fast_render, add_ground_plane=add_ground_plane,
            anchor=anchor, ground_normal=ground_normal,
            top_view=top_view, renderer=renderer)


def render_image_group(
        image_fn,
        camera_translation,
        vertices,
        camera_rotation,
        focal_length,
        camera_center,
        mesh_color,
        alpha=1.0,
        faces=None,
        mesh_filename= None,
        save_filename= None,
        keypoints_2d= None,
        track_ids = None,
        anchor=None, ground_normal=None,
        desc=[], fast_render=True, add_ground_plane=False,
        top_view=False, renderer='pyrender'
):
    try:
        image = cv2.imread(image_fn)[:,:,::-1]
    except:
        print(image_fn, 'does not exist')
        sys.exit(0)
        
    if np.max(image) > 10:
        image = image / 255.

    if keypoints_2d is not None:
        for kp_2d in keypoints_2d:
            image = draw_skeleton(image, kp_2d=kp_2d, dataset='openpose', unnormalize=False)

    # input image to this step should be between [0,1]
    camera_poses = [np.eye(4)]
    cam_dist_scalars = [1.0]
    if top_view:
        # view_rot = trimesh.transformations.rotation_matrix(
        #                 np.radians(45), [1, 0, 0])
        
        camera_pose = np.eye(4)
        rotation_matrix, translation_vector = look_at_camera(
            position=np.array([0, -3, -3]).astype(np.float32),
            target=np.array([0, 0, -4]).astype(np.float32),
            up=np.array([0, 1, 0]).astype(np.float32) # not true up, only used for calulating the right vector.
        )
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = translation_vector
        camera_poses.append(camera_pose)
        cam_dist_scalars.append(1.0)

    rendered_images = []
    for camera_pose, cam_dist_scalar in zip(camera_poses, cam_dist_scalars):
        rendered_img = render_with_pyrender(
            image=np.zeros_like(image),
            camera_translations=camera_translation,
            vertices=vertices,
            camera_rotation=camera_rotation,
            focal_length=focal_length,
            camera_center=camera_center,
            mesh_colors=mesh_color,
            alpha=alpha,
            faces=faces,
            mesh_filename=mesh_filename,
            camera_pose=camera_pose,
            cam_dist_scalar=cam_dist_scalar,
            anchor=anchor, ground_normal=ground_normal, fast_render=fast_render, add_ground_plane=add_ground_plane,
            renderer=renderer
        )
        rendered_images.append(rendered_img)
    output_img = np.concatenate([image, *rendered_images], axis=1)

    if save_filename is not None:
        images_save = output_img * 255
        images_save = np.clip(images_save, 0, 255).astype(np.uint8)
        # pad the top of images_save with white space
        images_save = np.concatenate([np.ones((100, images_save.shape[1], 3)).astype(np.uint8) * 255, images_save], axis=0)
        bottom_pixel = images_save.shape[0] + 30
        # pad the bottom of images_save with white space
        images_save = np.concatenate([images_save, np.ones((200, images_save.shape[1], 3)).astype(np.uint8) * 255], axis=0)
        
        for i, (color, kp_2d) in enumerate(zip(mesh_color, keypoints_2d)):
            R, G, B = list(colors[color])
            openpose_conf = int(round(kp_2d[:,2].mean(), 2) * 100)
            images_save = cv2.putText(
                images_save, f'Track id: {track_ids[i]}. OpenPose Conf: {openpose_conf} %', (10, 40+i*30), 
                cv2.FONT_HERSHEY_TRIPLEX, 1, (int(R), int(G), int(B)), 2)

        font_scale = min(images_save.shape[0], images_save.shape[1]) * 1e-3
        for i, text in enumerate(desc):
            images_save = cv2.putText(
                images_save, text, (10, bottom_pixel + i*40), 
                cv2.FONT_HERSHEY_TRIPLEX, font_scale, (120, 120, 120), 2)

        cv2.imwrite(save_filename, cv2.cvtColor(images_save, cv2.COLOR_BGR2RGB))

    return output_img


def render_with_pyrender(
        image: np.ndarray,
        camera_translations: list,
        vertices: list,
        camera_rotation: np.ndarray,
        focal_length: Tuple,
        camera_center: Tuple,
        mesh_colors: list,
        alpha: float = 1.0,
        faces: list = None,
        camera_pose: np.ndarray = None,
        cam_dist_scalar: float = 1.0,
        mesh_filename: str = None,
        add_ground_plane: bool = True,
        anchor=None, ground_normal=None,
        fast_render: bool=True, renderer='pyrender'
) -> np.ndarray:
    num_persons = len(vertices)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))

    for idx in range(num_persons):
        mesh_color = mesh_colors[idx]
        camera_translation = camera_translations[idx].copy()
        vertices_ = vertices[idx]
        faces_ = faces[idx]

        mesh_color = colors[mesh_color]
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(mesh_color[0] / 255., mesh_color[1] / 255., mesh_color[2] / 255., alpha))

        camera_translation[0] *= -1.
        # camera_translation[2] *= cam_dist_scalar
        # ground_translation = np.array([0, -ground_y, 10])
        # ground_translation[2] *= cam_dist_scalar

        mesh = trimesh.Trimesh(vertices_, faces_, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])  # invert y and z axis

        rot[:3, :3] = rot[:3, :3] #@ view_rot
        rot[:3, 3] = - camera_rotation @ camera_translation
        mesh.apply_transform(rot)

        if mesh_filename:
            mesh_filename_wo_ext, ext = os.path.splitext(mesh_filename)
            mesh.export(mesh_filename_wo_ext+'_'+str(idx)+ext)
            # if not mesh_filename.endswith('_rot.obj'):
            #     np.save(mesh_filename.replace('.obj', '.npy'), camera_translation)

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh, 'mesh'+str(idx))

    if add_ground_plane:
        ground_trimesh = get_checkerboard_plane(plane_width=10, num_boxes=10)
        pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])  # (x,y,z) -> (x,-z,y)
       
        ground_rot_mtx = rotation_matrix_between_vectors(np.array([0, 1, 0]), ground_normal)
        pose[:3, :3] = pose[:3, :3] @ ground_rot_mtx
        pose[:3, 3] = anchor

        for box in ground_trimesh:
            box.apply_transform(pose)
            
        if mesh_filename:
            combined_ground = trimesh.util.concatenate(ground_trimesh)
            mesh_filename_wo_ext, ext = os.path.splitext(mesh_filename)
            combined_ground.export(mesh_filename_wo_ext+'_ground'+ext)

        ground_mesh = pyrender.Mesh.from_trimesh(ground_trimesh, smooth=False)
        scene.add(ground_mesh, name='ground_plane')

    camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                       cx=camera_center[0], cy=camera_center[1])

    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    scene.add(light, pose=camera_pose)

    if not fast_render:  # turn on more lights -> slower but looks nicer.
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

    if renderer == 'pyrender':
        renderer = pyrender.OffscreenRenderer(
            viewport_width=image.shape[1],
            viewport_height=image.shape[0],
            point_size=1.0
        )
        try:
            color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        except:
            print('Error rendering image. Skipping...')
            return image
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)

    elif renderer == 'sim3drender':
        from vis_human.sim3drender import Sim3DR
        renderer = Sim3DR()
        cam_preds = torch.tensor(camera_translations).float().cuda()
        vertices = torch.tensor(vertices).float().cuda()
        
        verts, K = perspective_projection(vertices, cam_preds, focal_length=focal_length[0], camera_center=torch.tensor(camera_center).cuda())
        verts = torch.cat([verts, vertices[:,:,[2]]], -1)
        verts[:, :, 2] *= -1
        mesh_colors = np.array([colors[mesh_colors[idx]] for idx in range(num_persons)]) / 255.
        faces = faces[0].astype(np.int32)
        verts[:, :, 2] *= max(image.shape[:2])

        output_img = renderer(verts.cpu().numpy(), faces, (image*255).astype(np.uint8), mesh_colors=mesh_colors)
        output_img = output_img.astype(np.float32) / 255.

        # add bird's eye view
        verts_tran = vertices + cam_preds.unsqueeze(1)
        verts_bird_view, bbox3D_center, scale = rotate_view_weak_perspective(verts_tran, rx=-90, ry=0, img_shape=image.shape[:2], expand_ratio=1.2)
        output_img_sideview = renderer(verts_bird_view.cpu().numpy(), faces, (image*255).astype(np.uint8), mesh_colors=mesh_colors)
        output_img_sideview = output_img_sideview.astype(np.float32) / 255.
        # import pdb; pdb.set_trace()
        output_img = np.concatenate([output_img, output_img_sideview], axis=1)

        # cv2.imwrite("demo.png", output_img[:,:,::-1])
    return output_img

