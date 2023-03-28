# -*- coding: utf-8 -*-
import os
import argparse
import trimesh
# import mesh_to_sdf
from mesh_to_sdf import mesh_to_sdf
import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def save_pointcloud_to_ply(pnts: np.array, outfile: str):
    with open(outfile, 'w') as of:
        of.write('ply\n')
        of.write('format ascii 1.0\n')
        of.write(f'element vertex {pnts.shape[0]}\n')
        of.write('property float x\n')
        of.write('property float y\n')
        of.write('property float z\n')
        of.write('end_header\n')
        for i in range(pnts.shape[0]):
            of.write(f'{pnts[i][0]} {pnts[i][1]} {pnts[i][2]}\n')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--mesh",
        "-m",
        dest="mesh_path",
        required=True
    )
    arg_parser.add_argument(
        "--output",
        "-o",
        dest="output_path",
        required=True
    )
    arg_parser.add_argument(
        "--unify_center",
        "-c",
        dest="unify_center",
        default=False,
        action="store_true",
    )
    arg_parser.add_argument(
        "--unify_scale",
        "-s",
        dest="unify_scale",
        default=False,
        action="store_true",
    )
    args = arg_parser.parse_args()

    try:
        mesh = trimesh.load(args.mesh_path)
        pnts, sdfs = mesh_to_sdf.sample_sdf_near_surface(
            mesh,
            surface_point_method='sample',
            sphere_size=10.0
        )

        res = np.hstack([pnts, sdfs.reshape(-1, 1)])

        pos = res[res[:, 3] >= 0]
        neg = res[res[:, 3] < 0]

        save_pointcloud_to_ply(pos, 'pos.ply')
        save_pointcloud_to_ply(neg, 'neg.ply')

        np.savez(args.output_path, pos=pos, neg=neg)

    except Exception as e:
        raise f'error: {e}'
