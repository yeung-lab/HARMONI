""" Initialization and utility functions for SMPL body models. 
"""
import smplx
from .misc_utils import smpl_to_openpose, JointMapper

def init_body_model(model_path, batch_size, create_body_pose):
    """initialize SMPL models
    create_body_pose ([bool]): If use DAPA or VIBE initialization, then create_body_pose is True,
                               else, it is False.

    """
    joint_mapper = JointMapper(smpl_to_openpose('smpl', use_hands=False, use_face=False, 
                               use_face_contour=False, openpose_format='coco25'))
    model_params = dict(batch_size=batch_size, create_transl=True, joint_mapper=joint_mapper,
                        create_body_pose=create_body_pose)
    body_model = smplx.SMPL(model_path=model_path, **model_params)
    body_model.reset_params()
    body_model.transl.requires_grad = True
    return body_model
