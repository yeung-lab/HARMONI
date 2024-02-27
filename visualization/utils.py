import numpy as np
import trimesh
import torch


def get_colors():
    colors = {
        'pink': np.array([197, 27, 125]),  # L lower leg
        'light_pink': np.array([233, 163, 201]),  # L upper leg
        'light_green': np.array([161, 215, 106]),  # L lower arm
        'green': np.array([77, 146, 33]),  # L upper arm
        'red': np.array([215, 48, 39]),  # head
        'light_red': np.array([252, 146, 114]),  # head
        'light_orange': np.array([252, 141, 89]),  # chest
        'purple': np.array([118, 42, 131]),  # R lower leg
        'light_purple': np.array([175, 141, 195]),  # R upper
        'light_blue': np.array([145, 191, 219]),  # R lower arm
        'blue': np.array([69, 117, 180]),  # R upper arm
        'gray': np.array([130, 130, 130]),  #
        'white': np.array([255, 255, 255]),  #
        'pinkish': np.array([204, 77, 77]),
        'dark_blue': np.array((69,117,255)),
        'dark_green': np.array((21,89,5)),
    }
    return colors


def get_checkerboard_plane(plane_width=4, num_boxes=9, center=True):

    pw = plane_width / num_boxes
    white = [220, 220, 220, 100]
    black = [35, 35, 35, 100]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(
                center=[0, 0, -0.0001],
                extents=[pw, pw, 0.0002]
            )

            if center:
                c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i+j) % 2) == 0 else white
            meshes.append(ground)

    return meshes


def look_at_camera(position, target, up):
    forward = np.subtract(target, position)
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    rotation_matrix = np.array([
        [right[0], right[1], right[2]],
        [true_up[0], true_up[1], true_up[2]],
        [-forward[0], -forward[1], -forward[2]]
    ])

    translation_vector = -np.dot(rotation_matrix, position)

    return rotation_matrix, translation_vector


def rotation_matrix_between_vectors(v1, v2):
    v1 = np.array(v1)  # Convert to numpy array
    v2 = np.array(v2)

    v1_normalized = v1 / np.linalg.norm(v1)  # Normalize vectors
    v2_normalized = v2 / np.linalg.norm(v2)

    axis = np.cross(v1_normalized, v2_normalized)  # Calculate the rotation axis
    dot_product = np.dot(v1_normalized, v2_normalized)  # Calculate the dot product
    # import pdb; pdb.set_trace()
    skew_symmetric_matrix = np.array([[0, -axis[2], axis[1]],
                                      [axis[2], 0, -axis[0]],
                                      [-axis[1], axis[0], 0]])

    rotation_matrix = np.eye(3) + skew_symmetric_matrix + np.dot(skew_symmetric_matrix, skew_symmetric_matrix) * (1 / (1 + dot_product))

    return rotation_matrix


def perspective_projection(points, translation=None,rotation=None, keep_dim=False, 
                           focal_length=5000, camera_center=None):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    if isinstance(points,np.ndarray):
        points = torch.from_numpy(points).float()
    if isinstance(translation,np.ndarray):
        translation = torch.from_numpy(translation).float()
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    if camera_center is not None:
        K[:,-1, :-1] = camera_center

    # Transform points
    if rotation is not None:
        points = torch.einsum('bij,bkj->bki', rotation, points)
    if translation is not None:
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:,:,-1].unsqueeze(-1)+1e-4)
    if torch.isnan(points).sum()>0 or torch.isnan(projected_points).sum()>0:
       print('translation:', translation[torch.where(torch.isnan(translation))[0]])
       print('points nan value number:', len(torch.where(torch.isnan(points))[0]))

    # Apply camera intrinsics
    # projected_points = torch.einsum('bij,bkj->bki', K, projected_points)[:, :, :-1]
    projected_points = torch.matmul(projected_points.contiguous(), K.contiguous())
    if not keep_dim:
        projected_points = projected_points[:, :, :-1].contiguous()

    return projected_points, K


def get_rotate_x_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([
        [1, 0, 0], 
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]])
    return rot_mat

def get_rotate_y_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([
        [np.cos(angle), 0, np.sin(angle)], 
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]])
    return rot_mat

def rotate_view_weak_perspective(verts, rx=30, ry=0, img_shape=[512,512], expand_ratio=1.2, bbox3D_center=None, scale=None):
    device, dtype = verts.device, verts.dtype
    h, w = img_shape
    
    # front2birdview: rx=90, ry=0 ; front2sideview: rx=0, ry=90
    Rx_mat = get_rotate_x_mat(rx).type(dtype).to(device)
    Ry_mat = get_rotate_y_mat(ry).type(dtype).to(device)
    verts_rot = torch.einsum('bij,kj->bik', verts, Rx_mat)
    verts_rot = torch.einsum('bij,kj->bik', verts_rot, Ry_mat)
    
    if bbox3D_center is None:
        flatten_verts = verts_rot.view(-1, 3)
        # To move the vertices to the center of view, we get the bounding box of vertices and its center location 
        bbox3D_center = 0.5 * (flatten_verts.min(0).values + flatten_verts.max(0).values)[None, None]
    verts_aligned = verts_rot - bbox3D_center
    
    rendered_image_center = torch.Tensor([[[w / 2, h / 2]]]).to(device).type(verts_aligned.dtype)
    
    if scale is None:
        # To ensure all vertices are visible, we need to rescale the vertices
        scale = 1 / (expand_ratio * torch.abs(torch.div(verts_aligned[:,:,:2], rendered_image_center)).max()) 
    # move to the center of rendered image 
    verts_aligned *=  scale
    verts_aligned[:,:,:2] += rendered_image_center

    return verts_aligned, bbox3D_center, scale
