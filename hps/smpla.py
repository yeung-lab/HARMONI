import torch
import torch.nn as nn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'SMPL_Head':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23}

SMPL_Face_Foot_11 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28,
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34
}
SMPL_EXTRA_9 = {
    'R_Hip': 35, 'L_Hip':36, 'Neck_LSP':37, 'Head_top':38, 'Pelvis':39, 'Thorax_MPII':40, \
    'Spine_H36M':41, 'Jaw_H36M':42, 'Head':43}
OpenPose_25 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, 'L_Wrist':7, 'Mid_Hip':8, 'R_Hip':9, 'R_Knee':10, 'R_Ankle':11, \
    'L_Hip':12, 'L_Knee':13, 'L_Ankle':14, 'R_Eye':15, 'L_Eye':16, 'R_Ear':17, 'L_Ear':18, 'L_BigToe':19, 'L_SmallToe':20, 'L_Heel':21, 'R_BigToe':22, \
    'R_SmallToe':23, 'R_Heel':24
}
JOINT_NUM = 44
SMPL_ALL_44 = {**SMPL_24, **SMPL_Face_Foot_11, **SMPL_EXTRA_9}

def map_smpl_to_openpose(joints):
    openpose_joints = torch.zeros([len(joints), 25, 3], dtype=joints.dtype, device=joints.device)
    openpose_joints[:, OpenPose_25['Nose']] = joints[:, SMPL_ALL_44['Nose']]
    openpose_joints[:, OpenPose_25['Neck']] = joints[:, SMPL_ALL_44['Neck']]
    openpose_joints[:, OpenPose_25['R_Shoulder']] = joints[:, SMPL_ALL_44['R_Shoulder']]
    openpose_joints[:, OpenPose_25['R_Elbow']] = joints[:, SMPL_ALL_44['R_Elbow']]
    openpose_joints[:, OpenPose_25['R_Wrist']] = joints[:, SMPL_ALL_44['R_Wrist']]
    openpose_joints[:, OpenPose_25['L_Shoulder']] = joints[:, SMPL_ALL_44['L_Shoulder']]
    openpose_joints[:, OpenPose_25['L_Elbow']] = joints[:, SMPL_ALL_44['L_Elbow']]
    openpose_joints[:, OpenPose_25['L_Wrist']] = joints[:, SMPL_ALL_44['L_Wrist']]
    openpose_joints[:, OpenPose_25['Mid_Hip']] = joints[:, SMPL_ALL_44['Pelvis']]
    openpose_joints[:, OpenPose_25['R_Hip']] = joints[:, SMPL_ALL_44['R_Hip']]
    openpose_joints[:, OpenPose_25['R_Knee']] = joints[:, SMPL_ALL_44['R_Knee']]
    openpose_joints[:, OpenPose_25['R_Ankle']] = joints[:, SMPL_ALL_44['R_Ankle']]
    openpose_joints[:, OpenPose_25['L_Hip']] = joints[:, SMPL_ALL_44['L_Hip']]
    openpose_joints[:, OpenPose_25['L_Knee']] = joints[:, SMPL_ALL_44['L_Knee']]
    openpose_joints[:, OpenPose_25['L_Ankle']] = joints[:, SMPL_ALL_44['L_Ankle']]
    openpose_joints[:, OpenPose_25['R_Eye']] = joints[:, SMPL_ALL_44['R_Eye']]
    openpose_joints[:, OpenPose_25['L_Eye']] = joints[:, SMPL_ALL_44['L_Eye']]
    openpose_joints[:, OpenPose_25['R_Ear']] = joints[:, SMPL_ALL_44['R_Ear']]
    openpose_joints[:, OpenPose_25['L_Ear']] = joints[:, SMPL_ALL_44['L_Ear']]
    openpose_joints[:, OpenPose_25['L_BigToe']] = joints[:, SMPL_ALL_44['L_BigToe']]
    openpose_joints[:, OpenPose_25['L_SmallToe']] = joints[:, SMPL_ALL_44['L_SmallToe']]
    openpose_joints[:, OpenPose_25['L_Heel']] = joints[:, SMPL_ALL_44['L_Heel']]
    openpose_joints[:, OpenPose_25['R_BigToe']] = joints[:, SMPL_ALL_44['R_BigToe']]
    openpose_joints[:, OpenPose_25['R_SmallToe']] = joints[:, SMPL_ALL_44['R_SmallToe']]
    openpose_joints[:, OpenPose_25['R_Heel']] = joints[:, SMPL_ALL_44['R_Heel']]
    return openpose_joints


def regress_joints_from_vertices(vertices, J_regressor):
    if J_regressor.is_sparse:
        J = torch.stack([torch.sparse.mm(J_regressor, vertices[i]) for i in range(len(vertices))])
    else:
        J = torch.einsum('bik,ji->bjk', [vertices, J_regressor])
    return J

class VertexJointSelector(nn.Module):
    def __init__(self, extra_joints_idxs, J_regressor_extra9, J_regressor_h36m17, dtype=torch.float32):
        super(VertexJointSelector, self).__init__()
        self.register_buffer('extra_joints_idxs', extra_joints_idxs)
        if len(self.extra_joints_idxs) == 21:
            self.extra_joints_idxs = self.extra_joints_idxs[:11]
        self.register_buffer('J_regressor_extra9', J_regressor_extra9)
        self.register_buffer('J_regressor_h36m17', J_regressor_h36m17)

    def forward(self, vertices, joints):
        extra_joints11 = torch.index_select(vertices, 1, self.extra_joints_idxs)
        extra_joints9 = regress_joints_from_vertices(vertices, self.J_regressor_extra9)
        joints_h36m17 = regress_joints_from_vertices(vertices, self.J_regressor_h36m17)
        # 54 joints = 24 smpl joints + 21 face & feet & hands joints + 9 extra joints from different datasets + 17 joints from h36m
        joints44_17 = torch.cat([joints, extra_joints11, extra_joints9, joints_h36m17], dim=1)

        return joints44_17

class SMPL(nn.Module):
    def __init__(self, model_path, model_type='smpl', dtype=torch.float32):
        super(SMPL, self).__init__()
        self.dtype = dtype
        model_info = torch.load(model_path)

        self.faces = model_info['f']

        self.vertex_joint_selector = VertexJointSelector(model_info['extra_joints_index'], \
            model_info['J_regressor_extra9'], model_info['J_regressor_h36m17'], dtype=self.dtype)
        self.register_buffer('faces_tensor', model_info['f'])
        # The vertices of the template model
        self.register_buffer('v_template', model_info['v_template'])
        # The shape components, take the top 10 PCA componence.
        if model_type == 'smpl':
            self.register_buffer('shapedirs', model_info['shapedirs'])
        elif model_type == 'smpla':
            self.register_buffer('shapedirs', model_info['smpla_shapedirs'])
            
        self.register_buffer('J_regressor', model_info['J_regressor'])
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207, then transpose to 207 x 6890*3
        self.register_buffer('posedirs', model_info['posedirs'])
        # indices of parents for each joints
        self.register_buffer('parents', model_info['kintree_table'])
        self.register_buffer('lbs_weights',model_info['weights'])

    #@time_cost('SMPL')
    def forward(self, betas=None, poses=None, root_align=False, **kwargs):
        ''' Forward pass for the SMPL model
            Parameters
            ----------
            betas: torch.tensor, optional, shape Bx10
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            Return
            ----------
            outputs: dict, {'verts': vertices of body meshes, (B x 6890 x 3),
                            'joints54': 54 joints of body meshes, (B x 54 x 3), }
                            #'joints_h36m17': 17 joints of body meshes follow h36m skeleton format, (B x 17 x 3)}
        '''
        if isinstance(betas,np.ndarray):
            betas = torch.from_numpy(betas).type(self.dtype)
        if isinstance(poses,np.ndarray):
            poses = torch.from_numpy(poses).type(self.dtype)
        if poses.shape[-1] == 66:
            poses = torch.cat([poses, torch.zeros_like(poses[...,:6])], -1)

        default_device = self.shapedirs.device
        betas, poses = betas.to(default_device), poses.to(default_device)

        vertices, joints = lbs(betas, poses, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, dtype=self.dtype)
        joints44_17 = self.vertex_joint_selector(vertices, joints)

        if root_align:
            # use the Pelvis of most 2D image, not the original Pelvis
            root_trans = joints44_17[:,[SMPL_ALL_44['R_Hip'], SMPL_ALL_44['L_Hip']]].mean(1).unsqueeze(1)
            joints44_17 = joints44_17 - root_trans
            vertices =  vertices - root_trans

        return vertices, joints44_17


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = betas.shape[0]
    # Add shape contribution
    v_shaped = v_template + torch.einsum('bl,mkl->bmk', [betas, shapedirs])
    # Get the joints
    # NxJx3 array
    J = regress_joints_from_vertices(v_shaped, J_regressor)

    dtype = pose.dtype
    posedirs = posedirs.type(dtype)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=J_regressor.device)
    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3]).type(dtype)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1]).type(dtype)
    # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = torch.matmul(pose_feature, posedirs.type(dtype)) \
        .view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=J_regressor.device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


class SMPLA_parser(nn.Module):
    def __init__(self, smpla_path, smil_path, baby_thresh=0.8):
        super(SMPLA_parser, self).__init__()
        self.smil_model = SMPL(smil_path, model_type='smpl')
        self.smpla_model = SMPL(smpla_path, model_type='smpla')
        self.faces = self.smil_model.faces
        self.baby_thresh = baby_thresh
    
    def forward(self, betas=None, poses=None, transl=None, root_align=False, separate_smil_betas=False):
        baby_mask = betas[:,10] >= self.baby_thresh
        if baby_mask.sum()>0:
            adult_mask = ~baby_mask
            verts, joints = torch.zeros(len(poses), 6890, 3, device=poses.device, dtype=poses.dtype), torch.zeros(len(poses), JOINT_NUM+17, 3, device=poses.device, dtype=poses.dtype)

            # SMIL beta - 10 dims, only need the estimated betas, kid_offsets are not used
            if separate_smil_betas:
                verts[baby_mask], joints[baby_mask] = self.smil_model(betas=betas[baby_mask,11:], poses=poses[baby_mask], root_align=root_align)
            else:
                verts[baby_mask], joints[baby_mask] = self.smil_model(betas=betas[baby_mask,:10], poses=poses[baby_mask], root_align=root_align)
            
            # SMPLA beta - 11 dims, the estimated betas (10) + kid_offsets (1)
            if adult_mask.sum()>0:
                verts[adult_mask], joints[adult_mask] = self.smpla_model(betas=betas[adult_mask,:11], poses=poses[adult_mask], root_align=root_align)
        else:
            verts, joints = self.smpla_model(betas=betas[:,:11], poses=poses, root_align=root_align)

        if transl is not None:
            verts = verts + transl.unsqueeze(1)

        joints = map_smpl_to_openpose(joints)
        return verts, joints


def prepare_smil_model(dtype):
    model_path = 'data/body_models/smil_packed_info.pth'
    smpl_model = SMPL(model_path, dtype=dtype).eval()
    return smpl_model

def prepare_smpl_model(dtype, gender='neutral'):
    gender = gender.upper()
    model_path = f'data/body_models/smpl/SMPL_{gender}.pth'
    smpl_model = SMPL(model_path, dtype=dtype).eval()
    return smpl_model


def prepare_smpla_model(dtype, gender):
    gender = gender.upper()
    model_path = f'data/body_models/SMPLA_{gender}.pth'
    smpl_model = SMPLA_parser(model_path, smil_path='data/body_models/smil_packed_info.pth', baby_thresh=1.0).eval()
    return smpl_model


if __name__ == '__main__':
    import cv2
    from visualizer import Py3DR
    renderer = Py3DR()

    dtype = torch.float32
    # smpl_model = prepare_smpl_model(dtype)
    smpl_model = prepare_smpla_model(dtype, 'MALE')
   
    batch_size = 1
    a = torch.zeros([batch_size, 11]).type(dtype) + 2
    b = torch.zeros([batch_size, 72]).type(dtype)
    b[:, 0] = np.radians(180)

    image_length = 512
    bg_image = np.ones((image_length, image_length, 3), dtype=np.uint8) * 255

    rendered_images = []

    for age in np.linspace(0, 1, 10):
        a[:, -1] = age
        outputs = smpl_model(a, b)

        verts_np = outputs[0].cpu().numpy().astype(np.float32) + np.array([[[0., 0, 2]]]).astype(np.float32)
        height = verts_np[:, :, 1].max() - verts_np[:, :, 1].min()
        faces_np = smpl_model.faces.cpu().numpy().astype(np.int32)
        rendered_image, _ = renderer(verts_np, faces_np)

        # add text to image
        text = f'age coeff: {age:.2f}'
        text2 = f'height: {height:.2f} m'
        cv2.putText(rendered_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(rendered_image, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        rendered_images.append(rendered_image)

    rendered_image = np.concatenate(rendered_images, axis=1)
    cv2.imwrite('test.png', rendered_image)

    