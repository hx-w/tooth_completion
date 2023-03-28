# -*- coding: utf-8 -*-

from mesh_to_sdf import mesh_to_sdf

import trimesh

import numpy as np


neg_pnts = trimesh.load('neg_clipped.ply').vertices
pos_pnts = trimesh.load('pos_clipped.ply').vertices
mesh = trimesh.load('tooth_morphology/datasets/right_maxillary_central_incisor_outside/n31_zhaojinfeng/n31_zhaojinfeng.obj')

# pos_pnts = np.concatenate(pos_pnts).astype(np.float32)
# neg_pnts = np.concatenate(neg_pnts).astype(np.float32)
pos_sdfs = mesh_to_sdf.sample_sdf_with_queries(mesh, queries=pos_pnts, surface_point_method='sample').reshape(-1, 1)
neg_sdfs = mesh_to_sdf.sample_sdf_with_queries(mesh, queries=neg_pnts, surface_point_method='sample').reshape(-1, 1)

np.savez('n31_zhaojinfeng.npz', pos=np.hstack([pos_pnts, pos_sdfs]).astype(np.float32), neg=np.hstack([neg_pnts, neg_sdfs]).astype(np.float32))