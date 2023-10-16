# -*- coding: utf-8 -*-

import os
import trimesh
import numpy as np
import gradio as gr

# set package path
import sys
sys.path.append('.')
from mesh_to_sdf import mesh_to_sdf

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def sample_sdf_npz(mesh_path: str, target_path: str):
    try:
        mesh = trimesh.load(mesh_path)
        mesh.remove_unreferenced_vertices()
        mesh.remove_duplicate_faces()

        pnts, sdfs = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            surface_point_method='sample',
            sphere_size=11.0,
        )

        # print('==> sample finished', pnts.shape, sdfs.shape)

        datas = np.concatenate([pnts, sdfs.reshape(-1, 1)], axis=1)
        reserved_ = datas[datas[:, 2] < 0.0]
        clipped_ = datas[datas[:, 2] >= 0.0]
        np.random.shuffle(clipped_)
        clipped_ = clipped_[:int(clipped_.shape[0] / 1.5), :]
        datas = np.concatenate([clipped_, reserved_], axis=0)

        pnts = datas[:, :3]
        sdfs = datas[:, 3:]

        pntcloud = mesh_to_sdf.get_surface_point_cloud(
            mesh,
            surface_point_method='sample',
            sample_point_count=50000
        )

        surfs = np.concatenate([pntcloud.points, pntcloud.normals], axis=1)
        reserved_ = surfs[surfs[:, 2] < 0.0]
        clipped_ = surfs[surfs[:, 2] >= 0.0]
        np.random.shuffle(clipped_)
        clipped_ = clipped_[:int(clipped_.shape[0] / 1.5), :]
        surfs = np.concatenate([clipped_, reserved_], axis=0)
        surf_pnts = surfs[:, :3]
        surf_norms = surfs[:, 3:]

        res = np.hstack([pnts, sdfs.reshape(-1, 1)])
        res2 = np.hstack([surf_pnts, np.zeros((surf_pnts.shape[0], 1))])
        res = np.concatenate([res, res2], axis=0)

        pos = res[res[:, 3] >= 0]
        neg = res[res[:, 3] < 0]

        np.savez(target_path, pos=pos, neg=neg, surf_pnts=surf_pnts, surf_norms=surf_norms)

    except Exception as e:
        raise gr.Error(f'采样错误：{e}')
