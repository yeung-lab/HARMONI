import numpy as np
import trimesh


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
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

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
