#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import tqdm

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )

    if len(mesh_filenames) == 0:
        files = list(filter(lambda x: x.endswith('.obj'), os.listdir(shape_dir)))
        if len(files) == 0:
            raise NoMeshFileError()
        else:
            mesh_filenames = [os.path.join(shape_dir, files[0])]
            return mesh_filenames[0]
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    surf_pnts_tensor = torch.from_numpy(npz["surf_pnts"])
    surf_norms_tensor = torch.from_numpy(npz["surf_norms"])
    on_surfs = torch.cat([surf_pnts_tensor, surf_norms_tensor], 1)

    return [pos_tensor, neg_tensor, on_surfs]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]
    on_surfs = data[2]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    # on_surf_num = int(0 * subsample/ 8)
    # if on_surfs.shape[0] > on_surf_num:
    #     ind = random.randint(0, on_surfs.shape[0] - on_surf_num)
    #     sample_surf = on_surfs[ind : (ind + on_surf_num)]
    # else:
    #     random_surf = (torch.rand(on_surf_num) * on_surfs.shape[0]).long()
    #     sample_surf = torch.index_select(on_surfs, 0, random_surf)

    samples = torch.cat([sample_pos, sample_neg], 0)
    randidx = torch.randperm(samples.shape[0])
    samples = torch.index_select(samples, 0, randidx)

    # randidx = torch.randperm(sample_surf.shape[0])
    # sample_surf = torch.index_select(sample_surf, 0, randidx)
    
    # samples = torch.cat([samples, torch.zeros((sample_surf.shape[0], 4))], 0)
    # samples[-sample_surf.shape[0]:, :3] = sample_surf[:, :3]

    return {
        'coords': samples[:, :3],
        'sdfs': samples[:, 3:]
    }

    # same as DIF
    total_sample = samples.shape[0] + sample_surf.shape[0]
    coords = torch.cat([samples[:, :3], sample_surf[:, :3]], 0)
    # normals = torch.ones((total_sample, 3)) * -1
    # normals[samples.shape[0]:, ] = sample_surf[:, 3:]
    sdfs = torch.zeros((total_sample, 1))
    sdfs[:samples.shape[0], ] = samples[:, 3:]
    return {
        'coords': coords,
        'sdfs': sdfs,
        # 'normals': normals
    }

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in tqdm.tqdm(self.npyfiles, ascii=True):
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                surf_pnts_tensor = torch.from_numpy(npz["surf_pnts"])
                surf_norms_tensor = torch.from_numpy(npz["surf_norms"])
                on_surfs = torch.cat([surf_pnts_tensor, surf_norms_tensor], 1)
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        on_surfs[torch.randperm(on_surfs.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx
