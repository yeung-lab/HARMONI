import numpy as np
import torch
import torchvision.transforms as transforms
import scipy

TORSO_IDXS = [1,8,9,12,2,5]  # l/r shoulders, hip, neck, pelvis


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform_(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0]-1, pt[1]-1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)+1

def crop(img, center, scale, res, rot=0, edge_padding=True):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform_([1, 1], center, scale, res, invert=1))-1
    # Bottom right point
    br = np.array(transform_([res[0]+1, 
                             res[1]+1], center, scale, res, invert=1))-1
    
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    try:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], 
                                                            old_x[0]:old_x[1]]
    except:
        print(br, ul)
        print(old_x, old_y)
        print(new_x, new_y)
        print(img.shape, new_img.shape)
    
    if edge_padding:
        new_img[:new_y[0], new_x[0]:new_x[1]] = new_img[new_y[0], new_x[0]:new_x[1]]
        new_img[new_y[1]:, new_x[0]:new_x[1]] = new_img[new_y[1]-1, new_x[0]:new_x[1]]
        new_img[new_y[0]:new_y[1], :new_x[0]] = new_img[new_y[0]:new_y[1], np.newaxis, new_x[0]]
        new_img[new_y[0]:new_y[1], new_x[1]:] = new_img[new_y[0]:new_y[1], np.newaxis, new_x[1]-1]
        new_img[:new_y[0], :new_x[0]] = new_img[new_y[0], new_x[0]]
        new_img[new_y[1]:, :new_x[0]] = new_img[new_y[1]-1, new_x[0]]
        new_img[:new_y[0], new_x[1]:] = new_img[new_y[0], new_x[1]-1]
        new_img[new_y[1]:, new_x[1]:] = new_img[new_y[1]-1, new_x[1]-1]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = scipy.misc.imresize(new_img, res)
    return new_img


# make the torso to be one third of the image height.
def cropout_openpose_one_third(im_orig, pose, want_image=True):
    detected_joints = pose[pose[:, 2] > 0.0]
    x_min, x_max = detected_joints[:, 0].min(), detected_joints[:, 0].max()
    y_min, y_max = detected_joints[:, 1].min(), detected_joints[:, 1].max()
    center = [(x_min+x_max)/2, (y_min+y_max)/2] 
    torso_heights = []  # torso points: 2 (rshould), 9 (rhip), 5 (lshould), 12 (lhip)
    score_thres = 0.4
    if pose[9, 2]>score_thres and pose[2, 2]>score_thres:
        torso_heights.append(
            np.linalg.norm(pose[9,:2] - pose[2,:2]))
    if pose[5, 2]>score_thres and pose[12, 2]>score_thres:
        torso_heights.append(
            np.linalg.norm(pose[5,:2] - pose[12,:2]))
    if len(torso_heights) == 0:
        scale = max(x_max-x_min, y_max-y_min) * 1.2
    else:
        scale = max(np.mean(torso_heights) * 3, max(x_max-x_min, y_max-y_min) * 1.2)

    x = int(center[0]-scale//2)
    y = int(center[1]-scale//2)
    w = h = int(scale)
    crop_info = {'crop_boundary': {'y':y, 'h':scale, 'x':x, 'w':scale}}
    if want_image:
        cropped_image = crop(
            im_orig, center, scale, [224,224], edge_padding=True).astype(np.uint8)
        crop_info['cropped_image'] = cropped_image
    return crop_info


def cropout_openpose_torso(im_orig, pose, pad_factor=1.1):
    torso_idxs = [0,1,2,5,8,9,12,15,16,17,18]
    if pose[torso_idxs, 2].mean() > 0.4:
        pose = pose[torso_idxs]

    detected_joints = pose[pose[:, 2] > 0.0]
    x_min, x_max = detected_joints[:, 0].min(), detected_joints[:, 0].max()
    y_min, y_max = detected_joints[:, 1].min(), detected_joints[:, 1].max()
    center = [(x_min+x_max)/2, (y_min+y_max)/2] 
    
    scale = max(x_max-x_min, y_max-y_min) * pad_factor

    x = int(center[0]-scale//2)
    y = int(center[1]-scale//2)
    w = h = int(scale)
    crop_info = {'crop_boundary': {'y':y, 'h':scale, 'x':x, 'w':scale}}
    cropped_image = crop(
        im_orig, center, scale, [224,224], edge_padding=True).astype(np.uint8)
    crop_info['cropped_image'] = cropped_image
    return crop_info


def torso_area(kp_i):
    kp_i_torso = kp_i[TORSO_IDXS]
    kp_i_torso = kp_i_torso[kp_i_torso[:, 2] > 0, :2]
    if kp_i_torso.shape[0] == 0:
        kp_i_torso = kp_i[kp_i[:, 2] > 0, :2]
    area = np.prod(kp_i_torso.max(0) - kp_i_torso.min(0))
    return area


def default_img_transform(input_pil=False, size=224):
    trans_list = [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]    
    if not input_pil:
        trans_list = [transforms.ToPILImage()] + trans_list

    transform = transforms.Compose(trans_list)
    return transform


def unnormalize(normed):
    if len(normed.shape) == 3:
        unnormed = normed * torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        unnormed = unnormed + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    else:
        assert normed.shape[1] == 3
        unnormed = normed * torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        unnormed = unnormed + torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    return unnormed