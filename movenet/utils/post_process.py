from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def get_pred_depth(depth):
    return depth


def get_alpha(rot):
    # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # return rot[:, 0]
    idx = rot[:, 1] > rot[:, 5]
    alpha1 = np.arctan2(rot[:, 2], rot[:, 3]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6], rot[:, 7]) + (0.5 * np.pi)
    return alpha1 * idx + alpha2 * (1 - idx)


def ctdet_post_process(dets, c, s, h, w, num_classes):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = classes == j
            top_preds[j + 1] = np.concatenate(
                [
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32),
                ],
                axis=1,
            ).tolist()
        ret.append(top_preds)
    return ret


def multi_pose_post_process(dets, c, s, h, w):
    # dets: batch x max_dets x 40
    # return list of 39 in image coord
    ret = []
    for i in range(dets.shape[0]):
        bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
        pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
        top_preds = (
            np.concatenate(
                [bbox.reshape(-1, 4), dets[i, :, 4:5], pts.reshape(-1, 34)], axis=1
            )
            .astype(np.float32)
            .tolist()
        )
        ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
    return ret


def single_pose_post_process(dets, h, w):
    # post_process for Movenet especially
    # restore is the original size is not square.
    longEdge = max(h, w)
    dets[:, 0] = dets[:, 0] * longEdge
    dets[:, 1] = dets[:, 1] * longEdge
    if h > w:
        dets[:, 1] = dets[:, 1] - (h - w) // 2
    elif w > h:
        dets[:, 0] = dets[:, 0] - (w - h) // 2
    return dets
