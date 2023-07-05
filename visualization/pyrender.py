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
from utils.geometry import batch_euler2matrix
from visualization.utils import get_colors, get_checkerboard_plane

colors = get_colors()


def render(dataset, results, img_path, save_folder, cfg, cam_params, skip_if_no_infant=False, device='cuda', save_mesh=False,
           camera_center=np.array([960, 540]), img_list=None, fast_render=True, label_str=[''], use_smoothed=False,
           ground_y= -1.0, add_ground_plane=False, top_view=False):
    """Generate meshes from the saved estimated SMPL parameters, and then render.
    """
    adult_bm = init_body_model(cfg.smpl_model_path, batch_size=1, create_body_pose=True).to(device)
    infant_bm = init_body_model(cfg.smil_model_path, batch_size=1, create_body_pose=True).to(device)
    smpl_faces = adult_bm.faces

    # adult_feet = np.array([max(v) for v in results.adult_bottom.values() if len(v) > 0])
    # mean = np.mean(adult_feet)
    # std = np.std(adult_feet)
    # ground_y = np.mean(adult_feet[np.abs(adult_feet - mean) < 2 * std])
    ground_y = - 0.5
    logger.info('Ground plane is at '+str(ground_y))

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
        # print(img_name, 'has', len(persons), 'persons')
        if skip_if_no_infant and len(persons) == 1: continue

        # fetch ground information
        # adult_bottoms = results.adult_bottom[img_name]
        # if len(adult_bottoms) == 0:
        #     ground_y = mean_floor
        # else:
        #     ground_y = np.max(results.adult_bottom[img_name])
        #     if np.abs(ground_y-ground_y) > 0.5:
        #         ground_y = mean_floor

        if len(results.infant_bottom[img_name]) > 0:
            infant_to_ground = np.abs(ground_y - np.array(results.infant_bottom[img_name])).min()
            # desc = ['Infant to ground: {} m.'.format(round(infant_to_ground,3))] + label_str
            desc = label_str
        else:
            desc = label_str

        vertices = []
        faces = []
        mesh_colors = []
        keypoints_2d = []
        camera_translations = []
        body_types = []
        track_ids = []
        for person_id in persons:
            if person_id not in results.results: continue
            if use_smoothed:
                result = results.smoothed_results[person_id]
            else:
                result = results.results[person_id]
            is_ghost = dataset[person_id]['is_ghost']
            body_type = 'adult' if result['model_type'] == 'smplx' else 'infant'
            kp_2d = result['keypoints'][0,:25]
            pose, betas = to_tensor(result['body_pose']), to_tensor(result['betas'])
            global_orient = to_tensor(result['global_orient'])
            body_types.append(body_type)
            faces.append(smpl_faces)

            if result['model_type'] == 'smpl':
                bm = adult_bm(global_orient=global_orient, body_pose=pose, betas=betas)
                mesh_color = 'green'
            elif result['model_type'] == 'smil':
                bm = infant_bm(global_orient=global_orient, body_pose=pose, betas=betas)
                mesh_color = 'blue'
            else:
                raise Exception()

            if kp_2d is None or is_ghost:
                mesh_color = 'red'
            elif kp_2d[:,2].sum() > 9: 
                mesh_color = 'dark_'+mesh_color
            elif kp_2d[:,2].sum() < 5:
                mesh_color = 'light_'+mesh_color
            keypoints_2d.append(kp_2d)
            mesh_colors.append(mesh_color)
            camera_translations.append(result['transl'][0])
            vertices.append(to_numpy(bm.vertices)[0])
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
            faces=faces, cam_params=cam_params, keypoints_2d=keypoints_2d, track_ids=track_ids, save_filename=save_filename, ground_y=ground_y,
            mesh_color=mesh_colors, desc=desc, mesh_filename=mesh_filename, fast_render=fast_render, add_ground_plane=add_ground_plane,
            top_view=top_view)


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
        cam_params= None,
        ground_y= -1.0,
        desc=[], fast_render=True, add_ground_plane=False,
        top_view=False
):
    to_numpy = lambda x: x.detach().cpu().numpy()
    try:
        image = cv2.imread(image_fn)[:,:,::-1]
    except:
        print(image_fn, 'does not exist')
        sys.exit(0)
        
    if np.max(image) > 10:
        image = image / 255.

    # if keypoints_2d is not None:
    #     if isinstance(keypoints_2d, list):
    #         for kp_2d in keypoints_2d:
    #             image = draw_skeleton(image, kp_2d=kp_2d, dataset='openpose', unnormalize=False)
    #     else:
    #         image = draw_skeleton(image, kp_2d=keypoints_2d, dataset='openpose', unnormalize=False)

    # input image to this step should be between [0,1]
    view_rots = [np.eye(3)]
    cam_dist_scalars = [1.0]
    if top_view:
        view_rot = trimesh.transformations.rotation_matrix(
                        np.radians(45), [1, 0, 0])
        view_rots.append(view_rot[:3, :3])
        cam_dist_scalars.append(1.2)

    rendered_images = []
    for view_rot, cam_dist_scalar in zip(view_rots, cam_dist_scalars):
        rendered_img, ground_y = render_with_pyrender(
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
            view_rot=view_rot,
            cam_dist_scalar=cam_dist_scalar,
            ground_y=ground_y, fast_render=fast_render, add_ground_plane=add_ground_plane
        )
        rendered_images.append(rendered_img)
    output_img = np.concatenate([image, *rendered_images], axis=1)

    if save_filename is not None:
        images_save = output_img * 255
        images_save = np.clip(images_save, 0, 255).astype(np.uint8)
        # pad the top of images_save with white space
        images_save = np.concatenate([np.ones((120, images_save.shape[1], 3)).astype(np.uint8) * 255, images_save], axis=0)

        images_save = cv2.putText(
            images_save, 'Ground y: '+str(round(ground_y, 3)), (image.shape[1], 30), 
            cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
        
        for i, (color, kp_2d) in enumerate(zip(mesh_color, keypoints_2d)):
            R, G, B = list(colors[color])
            openpose_conf = round(kp_2d[:,2].mean(), 2) * 100
            images_save = cv2.putText(
                images_save, 'Track id: '+str(track_ids[i])+'. OpenPose Conf: '+str(openpose_conf) + "%", (image.shape[1], 70+i*30), 
                cv2.FONT_HERSHEY_TRIPLEX, 1, (int(R), int(G), int(B)), 2)
       
        for i, text in enumerate(desc):
            images_save = cv2.putText(
                images_save, text, (image.shape[1]+700, 20 + i*30), 
                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0),2)

        # images_save = cv2.resize(images_save, (1920, 540))
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
        view_rot: np.ndarray = None,
        cam_dist_scalar: float = 1.0,
        mesh_filename: str = None,
        add_ground_plane: bool = True,
        ground_y: float = -1.0,
        fast_render: bool=True
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
        camera_translation[2] *= cam_dist_scalar
        ground_translation = np.array([0, 0, 4.])
        ground_translation[2] *= cam_dist_scalar

        mesh = trimesh.Trimesh(vertices_, faces_, process=False)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])  # invert y and z axis

        rot[:3, :3] = rot[:3, :3] @ view_rot
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
        ground_trimesh = get_checkerboard_plane(plane_width=8)
        pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])  # (x,y,z) -> (x,-z,y)
        pose[:3, :3] = pose[:3, :3] @ view_rot
        pose[:3, 3] = - camera_rotation @ ground_translation
        pose[1, 3] = ground_y
        pose[0, 3] = 0

        for box in ground_trimesh:
            box.apply_transform(pose)
            
        if mesh_filename:
            combined_ground = trimesh.util.concatenate(ground_trimesh)
            mesh_filename_wo_ext, ext = os.path.splitext(mesh_filename)
            combined_ground.export(mesh_filename_wo_ext+'_ground'+ext)

        ground_mesh = pyrender.Mesh.from_trimesh(ground_trimesh, smooth=False)
        scene.add(ground_mesh, name='ground_plane')

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = camera_rotation
    camera = pyrender.IntrinsicsCamera(fx=focal_length[0], fy=focal_length[1],
                                       cx=camera_center[0], cy=camera_center[1])
    
    # if cam_dist_scalar > 1, move the camera a little bit farther so the scene is more clear
    # camera_pose[:3, 3] *= cam_dist_scalar

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

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image.shape[1],
        viewport_height=image.shape[0],
        point_size=1.0
    )

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:, :, None]
    output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
    return output_img, ground_y

