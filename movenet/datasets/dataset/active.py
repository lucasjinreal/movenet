from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data


class ACTIVE(data.Dataset):
    """
    The modified single-pose version of COCO human pose estimation dataset, naming `Active` dataset here. The main difference is that we limit the human counts in one single image to be less than or equal to 2.
    The order of joints:
        KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
        }
    """
    num_classes = 1
    num_joints = 17
    default_resolution = [192, 192] # mli: for movenet-lightning
    mean = np.array([1., 1., 1.],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([1., 1., 1.],
                   dtype=np.float32).reshape(1, 1, 3)
    flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                [11, 12], [13, 14], [15, 16]]

    def __init__(self, opt, split, sp=False):
        super(ACTIVE, self).__init__()
        self.edges = [[0, 1], [0, 2], [1, 3], [2, 4],
                      [4, 6], [3, 5], [5, 6],
                      [5, 7], [7, 9], [6, 8], [8, 10],
                      [6, 12], [5, 11], [11, 12],
                      [12, 14], [14, 16], [11, 13], [13, 15]]

        self.acc_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.data_dir = os.path.join(opt.data_dir, opt.dataset) # mli: the dir name is specified by `opt.dataset`
        self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
        if split == 'test':
            raise ValueError('No supported for the testing dataset.')
        else:
            self.annot_path = os.path.join(
                self.data_dir, 'annotations',
                '{}_{}.json').format(opt.dataset, split)
        self.max_objs = 2 # mli: only consider the images with less than 2 human objects
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        print('==> initializing {} {} data.'.format(opt.dataset, split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()

        if split == 'train':
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def bbox_from_kpt(self, kpts):
        bbox = np.zeros((4))
        xmin = np.min(kpts[:,0])
        ymin = np.min(kpts[:,1])
        xmax = np.max(kpts[:,0])
        ymax = np.max(kpts[:,1])
        width = xmax - xmin - 1
        height = ymax - ymin - 1
        
        # corrupted bounding box
        if width <= 0 or height <= 0:
            return bbox
        # 20% extend    
        else:
            bbox[0] = (xmin + xmax)/2. - width/2*1.2
            bbox[1] = (ymin + ymax)/2. - height/2*1.2
            bbox[2] = width*1.2
            bbox[3] = height*1.2
        return bbox

    def convert_eval_format(self, all_dets):
        # import pdb; pdb.set_trace()
        print()
        detections = []
        for image_id in all_dets:
            category_id = 1
            dets = all_dets[image_id]
            bbox = self.bbox_from_kpt(dets)
            bbox_out = list(map(self._to_float, bbox))
            score = np.sum(dets[:, 2]) / 4
            keypoints = np.concatenate([
                dets[:, [1, 0]],
                np.ones((17, 1), dtype=np.float32)], axis=1)
            keypoints[1:5] = np.zeros((4, 3))
            keypoints = keypoints.reshape(51).tolist()
            keypoints = list(map(self._to_float, keypoints))
            detection = {
                "image_id": int(image_id),
                "category_id": int(category_id),
                "bbox": bbox_out,
                "score": float("{:.2f}".format(score)),
                "keypoints": keypoints
            }
            detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
