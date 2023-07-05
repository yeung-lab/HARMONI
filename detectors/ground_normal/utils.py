import os, sys
import argparse

import cv2
from torchvision.transforms import Compose
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import trimesh
import pyransac3d as pyrsc

from detectors.midas.dpt_depth import DPTDepthModel
from detectors.midas.midas_net import MidasNet
from detectors.midas.midas_net_custom import MidasNet_small
from detectors.midas.transforms import Resize, NormalizeImage, PrepareForNet


pcd = o3d.geometry.PointCloud()

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def init_network(model_type, model_path, device, optimize):
    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize==True:
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()
    model.to(device)

    return model, net_w, net_h, resize_mode, normalization, transform


def estimate_plane_xy_diff_range(q_3d, xrange, yrange, name='plane.ply', return_normal=False,
                                 debug=False):
    # estimate plane points
    # range mins,maxs
    # bounds = -5.0
    # xy = np.arange(-bounds, bounds, 0.1)
    # l = len(xy)
    # xy_coords = np.repeat(xy, [l]).reshape([l, l])
    # coords_2d = np.stack([xy_coords, xy_coords.T], 2).reshape([-1, 2])

    plane1 = pyrsc.Plane()
    # Results in the plane equation Ax+By+Cz+D
    blockPrint()
    best_eq, best_inliers = plane1.fit(q_3d, 0.01)
    enablePrint()
    # in the form of a'x+b'y+c'z+d'=0. --> z = (-a'x- b'y -d') / c'
    a = best_eq[0]
    b = best_eq[1]
    c = best_eq[2]
    d = best_eq[3]

    normal_vect = np.array([a, b, c])

    if debug:
        x0 = xrange[0]
        x1 = xrange[1]
        # xran = x1-x0
        y0 = yrange[0]
        y1 = yrange[1]
        # yran = y1-y0

        resolution = 100
        xx = np.linspace(x0, x1, num=resolution)
        yy = np.linspace(y0, y1, num=resolution)
        X, Y = np.meshgrid(xx, yy)
        coords_2d = np.stack([X, Y], 2).reshape([-1, 2])

        A = -1 * a / c
        B = -1 * b / c
        C = -1 * d / c
        z = A * coords_2d[:, 0] + B * coords_2d[:, 1] + C
        ground_est = np.concatenate([coords_2d, z[:, None]], 1)

        # ones = np.ones_like(z[:, None] )
        # xyin = np.concatenate([coords_2d, ones], 1)
        # save_points(xyin, './results/xy_points.ply')

        pc = trimesh.PointCloud(ground_est)
        pcd.points = o3d.utility.Vector3dVector(pc)
        # fname = "./results/estimated_ground_%s.ply"%(name)
        o3d.io.write_point_cloud(name, pcd)
        # print('plane saved at: %s' % name)

    if return_normal:
        return normal_vect


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def write_depth(path, depth, bits=1, save=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)

    if save:
        write_pfm(path + ".pfm", depth.astype(np.float32))
        if bits == 1:
            cv2.imwrite(path + ".png", out.astype("uint8"))
        elif bits == 2:
            cv2.imwrite(path + ".png", out.astype("uint16"))

    return out


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img