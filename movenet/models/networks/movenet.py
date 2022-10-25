# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# Modified by Min Li
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
from turtle import shape

import cv2

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from .backbone_utils import mobilenet_backbone
import torch.utils.model_zoo as model_zoo
from alfred import logger, print_shape

BN_MOMENTUM = 0.1


class MoveNet(nn.Module):
    """
    MoveNet from Goolge. Please refer their blog: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html

    """

    def __init__(self, backbone, heads, head_conv, ft_size=48):
        super(MoveNet, self).__init__()
        self.num_joints = heads["hm_hp"]
        self.out_channels = 24
        self.backbone = backbone
        self.heads = heads
        self.ft_size = ft_size
        self.weight_to_center = self._generate_center_dist(self.ft_size).unsqueeze(2)

        logger.info(f"{self.heads}, ft_size: {self.ft_size}")

        self.dist_y, self.dist_x = self._generate_dist_map(self.ft_size)
        self.index_17 = torch.arange(0, self.num_joints).float()

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(
                        self.out_channels,
                        self.out_channels,
                        3,
                        padding=1,
                        groups=self.out_channels,
                        bias=True,
                    ),
                    nn.Conv2d(self.out_channels, head_conv, 1, 1, 0, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        head_conv,
                        classes,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True,
                    ),
                )
            else:
                fc = nn.Conv2d(
                    64, classes, kernel_size=1, stride=1, padding=0, bias=True
                )
            self.__setattr__(head, fc)

    def forward(self, x):
        # conv forward
        # x  = x * 0.007843137718737125 - 1.0
        # specify the device
        device = x.device
        self.weight_to_center = self.weight_to_center.to(device)
        self.dist_y, self.dist_x = self.dist_y.to(device), self.dist_x.to(device)

        x = self.backbone(x)  # backbone is right
        # return x
        ret = {}
        print(self.heads)
        # return x
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        if torch.jit.is_tracing():
            logger.info("in tracing mode....")
            hm = ret["hm"]
            hm_hp = ret["hm_hp"]
            hps = ret["hps"]
            hp_offset = ret["hp_offset"]
            print_shape(hm, hm_hp, hps, hp_offset)
            ret = self.decode_jit(hm_hp, hm, hps, hp_offset)
            return ret
        else:
            return [ret]

    def decode_jit(self, hm_hp, hm, hps, hp_offset):
        kpt_heatmap, center, kpt_regress, kpt_offset = (
            hm_hp.squeeze(0).permute((1, 2, 0)),
            hm.squeeze(0).permute((1, 2, 0)),
            hps.squeeze(0).permute((1, 2, 0)),
            hp_offset.squeeze(0).permute((1, 2, 0)),
        )
        # return kpt_heatmap
        # pose decode
        kpt_heatmap = torch.sigmoid(kpt_heatmap)
        # return kpt_heatmap
        center = torch.sigmoid(center)
        print_shape(center)
        ct_ind = self._top_with_center(center)
        print_shape(kpt_regress)
        kpt_coor = self._center_to_kpt(kpt_regress, ct_ind)
        # return kpt_coor
        kpt_top_inds = self._kpt_from_heatmap(kpt_heatmap, kpt_coor)
        # return kpt_top_inds
        kpt_with_conf = self._kpt_from_offset(
            kpt_offset, kpt_top_inds, kpt_heatmap, self.ft_size
        )
        return kpt_with_conf

    def decode(self, x):
        kpt_heatmap, center, kpt_regress, kpt_offset = (
            x["hm_hp"].squeeze(0).permute((1, 2, 0)),
            x["hm"].squeeze(0).permute((1, 2, 0)),
            x["hps"].squeeze(0).permute((1, 2, 0)),
            x["hp_offset"].squeeze(0).permute((1, 2, 0)),
        )
        # pose decode
        kpt_heatmap = torch.sigmoid(kpt_heatmap)
        center = torch.sigmoid(center)
        ct_ind = self._top_with_center(center)
        kpt_coor = self._center_to_kpt(kpt_regress, ct_ind)
        kpt_top_inds = self._kpt_from_heatmap(kpt_heatmap, kpt_coor)
        kpt_with_conf = self._kpt_from_offset(
            kpt_offset, kpt_top_inds, kpt_heatmap, self.ft_size
        )
        return kpt_with_conf

    def _draw(self, ft):
        plt.imshow(ft.numpy().reshape(self.ft_size, self.ft_size))
        plt.show()

    def _generate_center_dist(self, ft_size=48, delta=1.8):
        weight_to_center = torch.zeros((int(ft_size), int(ft_size)))
        y, x = np.ogrid[0:ft_size, 0:ft_size]
        center_y, center_x = ft_size / 2.0, ft_size / 2.0
        y = y - center_y
        x = x - center_x
        weight_to_center = 1 / (np.sqrt(y * y + x * x) + delta)
        weight_to_center = torch.from_numpy(weight_to_center).to(torch.float32)
        return weight_to_center

    def _generate_dist_map(self, ft_size=48):
        y, x = np.ogrid[0:ft_size, 0:ft_size]
        y = torch.from_numpy(np.repeat(y, ft_size, axis=1)).unsqueeze(2).float()
        x = torch.from_numpy(np.repeat(x, ft_size, axis=0)).unsqueeze(2).float()
        return y, x

    def _top_with_center(self, center):
        scores = center * self.weight_to_center
        top_ind = torch.argmax(scores.view(1, self.ft_size * self.ft_size, 1), dim=1)
        return top_ind

    def _center_to_kpt(self, kpt_regress, ct_ind, ft_size=48):
        # we need implicit specific floor op here for wnn support
        ct_y = torch.div(ct_ind, ft_size)
        ct_y = torch.floor(ct_y)
        print(ct_y)
        # return ct_y
        ct_x = ct_ind - ct_y * ft_size
        print_shape(ct_x, ct_y)
        print(ct_x)
        # return ct_x
        kpt_regress = kpt_regress.view(-1, self.num_joints, 2)
        print_shape(ct_ind)
        ct_ind = ct_ind.unsqueeze(2).repeat(ct_ind.size(0), self.num_joints, 2)
        print_shape(ct_ind)
        # return ct_ind
        kpt_coor = kpt_regress.gather(0, ct_ind).squeeze(0)
        kpt_coor = kpt_coor + torch.cat((ct_y, ct_x), dim=1)
        return kpt_coor

    def _kpt_from_heatmap(self, kpt_heatmap, kpt_coor):
        y = self.dist_y - kpt_coor[:, 0].reshape(1, 1, self.num_joints)
        x = self.dist_x - kpt_coor[:, 1].reshape(1, 1, self.num_joints)
        dist_weight = torch.sqrt(y * y + x * x) + 1.8

        scores = kpt_heatmap / dist_weight
        # return scores
        scores = scores.reshape((1, self.ft_size * self.ft_size, self.num_joints))
        top_inds = torch.argmax(scores, dim=1)
        return top_inds

    def _kpt_from_offset(self, kpt_offset, kpt_top_inds, kpt_heatmap, size=48):
        kpts_ys = torch.div(kpt_top_inds, size)
        kpts_ys = torch.floor(kpts_ys)
        kpts_xs = kpt_top_inds - kpts_ys * size

        kpt_coordinate = torch.concat([kpts_ys, kpts_xs], dim=0).permute(1, 0)
        print_shape(kpt_coordinate, kpts_ys, kpts_xs)
        # return kpt_coordinate

        kpt_heatmap = kpt_heatmap.view(-1, self.num_joints)
        kpt_conf = kpt_heatmap.gather(0, kpt_top_inds).squeeze(0)

        kpt_offset = kpt_offset.view(-1, self.num_joints, 2)

        kpt_top_inds = kpt_top_inds.unsqueeze(-1)
        kpt_top_inds = torch.concat([kpt_top_inds, kpt_top_inds], dim=-1)
        # kpt_top_inds = kpt_top_inds.unsqueeze(2).expand(
        #     kpt_top_inds.size(0), self.num_joints, 2
        # )
        kpt_offset_yx = kpt_offset.gather(0, kpt_top_inds).squeeze(0)
        kpt_coordinate = (kpt_offset_yx + kpt_coordinate) / size
        kpt_with_conf = torch.cat(
            [kpt_coordinate, kpt_conf.unsqueeze(1)], dim=1
        ).reshape((1, 1, self.num_joints, 3))
        return kpt_with_conf


# def get_pose_net(heads, head_conv=96, froze_backbone=True):
#     backbone = mobilenet_backbone('mobilenet_v2', pretrained=False, fpn=True)
#     if froze_backbone:
#         for param in backbone.parameters():
#             param.requires_grad = False
#     model = MoveNet(backbone, heads, head_conv=head_conv)
#     return model


def get_pose_net(heads, head_conv=96, froze_backbone=True, model_type="lighting"):
    logger.info(f"---- model_type: {model_type}")
    backbone = mobilenet_backbone(
        "mobilenet_v2",
        pretrained=False,
        fpn=True,
        trainable_layers=0,
        model_type=model_type,
    )
    if froze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    if model_type == "lighting":
        # ft_size = 48
        ft_size = 64
    else:
        ft_size = 64
    logger.info(f"{ft_size}, {model_type}")
    model = MoveNet(backbone, heads, head_conv=head_conv, ft_size=ft_size)
    # froze
    """for k,v in model.named_parameters():
        head_name = k.split('.')[0]
        if head_name == 'hm' or head_name == 'hps':
            v.requires_grad = False"""

    return model
