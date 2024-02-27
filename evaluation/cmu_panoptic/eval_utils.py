"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
import os
import json
import yaml
import cv2
import torch
import numpy as np
from loguru import logger
# from .geometry import euler_angles_from_rotmat
SMPL_OR_JOINTS = np.array([0, 1, 2, 4, 5, 16, 17, 18, 19])


def joint_angle_error(pred_mat, gt_mat):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 24, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 24, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """

    gt_mat = gt_mat[:, SMPL_OR_JOINTS, :, :]
    pred_mat = pred_mat[:, SMPL_OR_JOINTS, :, :]

    # Reshape the matrices into B x 3 x 3 arrays
    r1 = np.reshape(pred_mat, [-1, 3, 3])
    r2 = np.reshape(gt_mat, [-1, 3, 3])

    # Transpose gt matrices
    r2t = np.transpose(r2, [0, 2, 1])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(r1, r2t)

    angles = []
    # Convert rotation matrix to axis angle representation and find the angle
    for i in range(r1.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))

    return np.mean(np.array(angles))



def compute_similarity_transform_pitchyawroll(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    
    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t


    import math
    from .geometry import batch_euler2matrix
    s1, s2 = euler_angles_from_rotmat(torch.tensor(R.T).unsqueeze(0))
    pitch = -math.radians(s1[0])
    yaw = -math.radians(s1[1])
    roll = -math.radians(s1[2])

    R_pitch = batch_euler2matrix(torch.tensor([pitch,0,0])).squeeze().cpu().numpy()
    R_yaw = batch_euler2matrix(torch.tensor([0,yaw,0])).squeeze().cpu().numpy()
    R_roll = batch_euler2matrix(torch.tensor([0,0,roll])).squeeze().cpu().numpy()

    t = mu2 - scale*(R_pitch.dot(mu1))
    S1_pitch = scale*R_pitch.dot(S1) + t


    t = mu2 - scale*(R_yaw.dot(mu1))
    S1_yaw = scale*R_yaw.dot(S1) + t


    t = mu2 - scale*(R_roll.dot(mu1))
    S1_roll = scale*R_roll.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T
        S1_pitch = S1_pitch.T
        S1_yaw = S1_yaw.T
        S1_roll = S1_roll.T

    return S1_hat, S1_pitch, S1_yaw, S1_roll


def compute_similarity_transform_rotation(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return R, t

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    
    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))


    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def compute_similarity_transform_batch_pitchyawroll(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    S1_pitch = np.zeros_like(S1)
    S1_yaw = np.zeros_like(S1)
    S1_roll = np.zeros_like(S1)

    for i in range(S1.shape[0]):
        S1_hat[i], S1_pitch[i], S1_yaw[i], S1_roll[i] = compute_similarity_transform_pitchyawroll(S1[i], S2[i])
    return S1_hat, S1_pitch, S1_yaw, S1_roll

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)

    re_per_joint = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))
    re = re_per_joint
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    else:
        re = re
    return re, re_per_joint, S1_hat


def reconstruction_error_pitchyawroll(S1, S2):
    """Do Procrustes alignment and compute reconstruction error."""

    pred_hat, pred_pitch, pred_yaw, pred_roll = compute_similarity_transform_batch_pitchyawroll(S1, S2)
    pred_hat_err = np.sqrt( ((pred_hat - S2)** 2).sum(axis=-1))
    pred_pitch_err = np.sqrt( ((pred_pitch - S2)** 2).sum(axis=-1))
    pred_yaw_err = np.sqrt( ((pred_yaw - S2)** 2).sum(axis=-1))
    pred_roll_err = np.sqrt( ((pred_roll - S2)** 2).sum(axis=-1))

    return pred_hat_err, pred_pitch_err, pred_yaw_err, pred_roll_err
   
