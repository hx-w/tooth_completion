# -*- coding: utf-8 -*-

import json
import os
import random
import torch
import trimesh
import numpy as np
from tqdm import tqdm
import gradio as gr

import sys
sys.path.append('.')

# os.environ['PYOPENGL_PLATFORM'] = 'egl'
from mesh_to_sdf import mesh_to_sdf

import deep_sdf
import deep_sdf.workspace as ws


def reconstruct_latent(npz_name, iters, exp_model):
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    specs_filename = os.path.join(exp_model, "specs.json")

    specs = json.load(open(specs_filename, "r"))

    saved_model_state = torch.load(
        os.path.join(
            exp_model, ws.model_params_subdir, "latest.pth"
        )
    )
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module.cuda()

    clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    data_sdf = deep_sdf.data.read_sdf_samples_into_ram(npz_name)

    data_sdf[0] = data_sdf[0][torch.randperm(data_sdf[0].shape[0])]
    data_sdf[1] = data_sdf[1][torch.randperm(data_sdf[1].shape[0])]

    ## Start Reconstruction
    def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every, iter
    ):
        lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


    decreased_by = 10
    stat = 0.01
    clamp_dist = 1.0
    num_samples = 8000
    lr = 5e-3
    l2reg = True

    adjust_lr_every = int(iters / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat).cuda()
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach()).cuda()

    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    for e in tqdm(range(iters), desc='重建符号距离场'):
        decoder.eval()
        sdf_data = deep_sdf.data.unpack_sdf_samples_from_ram(
            data_sdf, num_samples
        )
        xyz = sdf_data['coords'].cuda()
        sdf_gt = sdf_data['sdfs'].cuda().unsqueeze(1)

        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every, e)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)

        inputs = torch.cat([latent_inputs, xyz], 1).cuda().to(torch.float32)

        pred_sdf = decoder(inputs)

        if e == 0:
            pred_sdf = decoder(inputs)

        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)

        loss = loss_l1(pred_sdf, sdf_gt.reshape(pred_sdf.shape))
        if l2reg:
            loss += 1e-3 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        loss_num = loss.item()

    return loss_num, latent, decoder


def extract_mesh_from_latent(decoder, latent, mesh_path, resols):
    try:
        deep_sdf.mesh.create_slice_heatmap(decoder, latent, mesh_path+"_XOZ.png", 22, 256, None, 0, None)
        deep_sdf.mesh.create_slice_heatmap(decoder, latent, mesh_path+"_YOZ.png", 22, 256, 0, None, None)
        deep_sdf.mesh.create_slice_heatmap(decoder, latent, mesh_path+"_XOY.png", 22, 256, None, None, 0)

        deep_sdf.mesh.create_mesh(decoder, latent, mesh_path, resols, max_batch=int(2 ** 16), volume_size=20)
        raw_mesh = trimesh.load(mesh_path + '.ply')
        res = sorted(
            raw_mesh.split(only_watertight=False),
            key=lambda x: x.vertices.shape[0], reverse=True
        )[0]
        res.remove_unreferenced_vertices()
        res.remove_degenerate_faces()
        res.remove_infinite_values()
        res.remove_duplicate_faces()
        res.vertex_normals
        res.export(mesh_path + '.obj', include_normals=True)

        os.remove(mesh_path + '.ply')

        return [mesh_path + '_XOZ.png', mesh_path + '_YOZ.png', mesh_path + '_XOY.png']

    except Exception as e:
        raise gr.Error(f'提取网格错误：{e}')


def compute_mesh_errors(tgt_mesh, ref_mesh):
    tgt_mesh = trimesh.load(tgt_mesh)
    ref_mesh = trimesh.load(ref_mesh)

    dists = mesh_to_sdf.sample_sdf_with_queries(
        tgt_mesh,
        ref_mesh.vertices,
        surface_point_method='sample',
        sample_point_count=1000000
    )
    # dists = tgt_mesh.nearest.signed_distance(ref_mesh.vertices)
    max_dist = np.max(dists)
    min_dist = np.min(dists)

    x_axis = np.linspace(min_dist, max_dist, 100)
    y_axis = np.zeros_like(x_axis)

    for i, x in tqdm(enumerate(x_axis), desc='计算误差分布'):
        y_axis[i] = len(dists[np.logical_and(dists >= x, dists < x + (max_dist - min_dist) / 100)])

    return x_axis, y_axis
