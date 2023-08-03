import ssl
import shutil
import wget
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import json
import numpy as np
import h5py
import glob
from utils import data_augmentation
import pickle


# ================================================================================
# AnTao350M shapenet dataloader

def download_shapenet_AnTao350M(url, saved_path):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    if not os.path.exists(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')):
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('mv %s %s' % ('hdf5_data', os.path.join(saved_path, 'shapenet_part_seg_hdf5_data')))
        os.system('rm %s' % (zipfile))


class ShapeNet_AnTao350M(torch.utils.data.Dataset):
    def __init__(self, saved_path, partition, selected_points, augmentation, num_aug, jitter, std, clip, rotate,
                 which_axis,
                 angle_range, translate, x_translate_range, y_translate_range, z_translate_range, anisotropic_scale,
                 x_scale_range, y_scale_range, z_scale_range):
        self.selected_points = selected_points
        self.augmentation = augmentation
        self.num_aug = num_aug
        if augmentation:
            self.augmentation_list = []
            if jitter:
                self.augmentation_list.append([data_augmentation.jitter, [std, clip]])
            if rotate:
                self.augmentation_list.append([data_augmentation.rotate, [which_axis, angle_range]])
            if translate:
                self.augmentation_list.append(
                    [data_augmentation.translate, [x_translate_range, y_translate_range, z_translate_range]])
            if anisotropic_scale:
                self.augmentation_list.append(
                    [data_augmentation.anisotropic_scale, [x_scale_range, y_scale_range, z_scale_range]])
            if not jitter and not rotate and not translate and not anisotropic_scale:
                raise ValueError('At least one kind of data augmentation should be applied!')
            if len(self.augmentation_list) < num_aug:
                raise ValueError(
                    f'num_aug should not be less than the number of enabled augmentations. num_aug: {num_aug}, number of enabled augmentations: {len(self.augmentation_list)}')
        self.all_pcd = []
        self.all_cls_label = []
        self.all_seg_label = []
        if partition == 'trainval':
            file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
                   + glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
        else:
            file = glob.glob(os.path.join(saved_path, 'shapenet_part_seg_hdf5_data', '*%s*.h5' % partition))
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            pcd = f['data'][:].astype('float32')
            cls_label = f['label'][:].astype('int64')
            seg_label = f['pid'][:].astype('int64')
            f.close()
            self.all_pcd.append(pcd)
            self.all_cls_label.append(cls_label)
            self.all_seg_label.append(seg_label)
        self.all_pcd = np.concatenate(self.all_pcd, axis=0)
        self.all_cls_label = np.concatenate(self.all_cls_label, axis=0)
        self.all_seg_label = np.concatenate(self.all_seg_label, axis=0)

    def __len__(self):
        return self.all_cls_label.shape[0]

    def __getitem__(self, index):
        # get category one hot
        category_id = self.all_cls_label[index, 0]
        category_onehot = F.one_hot(torch.Tensor([category_id]).long(), 16).to(torch.float32).permute(1, 0)

        # get point cloud
        pcd = self.all_pcd[index]
        if self.augmentation:
            choice = np.random.choice(len(self.augmentation_list), self.num_aug, replace=False)
            for i in choice:
                augmentation, params = self.augmentation_list[i]
                pcd = augmentation(pcd, *params)
        pcd = torch.Tensor(pcd).to(torch.float32)
        pcd = pcd.permute(1, 0)

        # get point cloud seg label
        seg_label = self.all_seg_label[index].astype('float32')
        # match parts id and convert seg label to one hot
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot

        # get point cloud seg label
        seg_label = self.all_seg_label[index].astype('float32')
        seg_label = seg_label[indices]
        # match parts id and convert seg label to one hot
        seg_label = F.one_hot(torch.Tensor(seg_label).long(), 50).to(torch.float32).permute(1, 0)

        # pcd.shape == (3, N)    seg_label.shape == (50, N)    category_onehot.shape == (16, 1)
        return pcd, seg_label, category_onehot


def get_shapenet_dataset_AnTao350M(saved_path, selected_points, augmentation, num_aug, jitter, std, clip, rotate,
                                   which_axis,
                                   angle_range, translate, x_translate_range, y_translate_range, z_translate_range,
                                   anisotropic_scale,
                                   x_scale_range, y_scale_range, z_scale_range):
    # get dataset
    train_set = ShapeNet_AnTao350M(saved_path, 'train', selected_points, augmentation, num_aug, jitter, std, clip,
                                   rotate, which_axis,
                                   angle_range, translate, x_translate_range, y_translate_range, z_translate_range,
                                   anisotropic_scale,
                                   x_scale_range, y_scale_range, z_scale_range)
    validation_set = ShapeNet_AnTao350M(saved_path, 'val', selected_points, False, num_aug, jitter, std, clip, rotate,
                                        which_axis,
                                        angle_range, translate, x_translate_range, y_translate_range, z_translate_range,
                                        anisotropic_scale,
                                        x_scale_range, y_scale_range, z_scale_range)
    trainval_set = ShapeNet_AnTao350M(saved_path, 'trainval', selected_points, augmentation, num_aug, jitter, std, clip,
                                      rotate, which_axis,
                                      angle_range, translate, x_translate_range, y_translate_range, z_translate_range,
                                      anisotropic_scale,
                                      x_scale_range, y_scale_range, z_scale_range)
    test_set = ShapeNet_AnTao350M(saved_path, 'test', selected_points, False, num_aug, jitter, std, clip, rotate,
                                  which_axis,
                                  angle_range, translate, x_translate_range, y_translate_range, z_translate_range,
                                  anisotropic_scale,
                                  x_scale_range, y_scale_range, z_scale_range)
    return train_set, validation_set, trainval_set, test_set

# ================================================================================
