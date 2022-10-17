from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger


class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads,
                                  opt.head_conv, opt.froze_backbone)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()
        
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.opt = opt
        self.pause = True
        self.global_num = 0

    def pre_process(self, image, meta=None):
        height, width = image.shape[0:2]

        # padding all images to be square.
        if height > width:
            diff = height - width
            image = cv2.copyMakeBorder(
                image, 0, 0, int(diff//2), int(diff//2 + diff%2),
                cv2.BORDER_CONSTANT, value=(0,0,0))
        elif height < width:
            diff = width - height
            image = cv2.copyMakeBorder(
                image, int(diff//2), int(diff//2+diff%2), 0, 0,
                cv2.BORDER_CONSTANT, value=(0,0,0))

        new_height = 256#192
        new_width = 256#192

        inp_height = new_height
        inp_width = new_width
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

        inp_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        inp_image = cv2.cvtColor(inp_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp_image = ((inp_image / 127.5 - self.mean) /
                     self.std).astype(np.float32)
        images = inp_image.transpose(2, 0, 1).reshape(
            1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'in_height': height,
                'in_width': width,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        return images, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        # detections = []
        scale_start_time = time.time()

        images, meta = self.pre_process(image, meta)

        images = images.to(self.opt.device)
        # torch.cuda.synchronize()
        pre_process_time = time.time()
        pre_time += pre_process_time - scale_start_time
        output, dets, forward_time = self.process(images, return_time=True)
        # torch.cuda.synchronize()
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time
        if self.opt.debug >= 2:
            self.debug(debugger, images, dets, output)
        dets = self.post_process(dets, meta)
        # torch.cuda.synchronize()
        post_process_time = time.time()
        post_time += post_process_time - decode_time
        results = dets

        # results = self.merge_outputs(detections)
        # torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results, prefix=self.global_num)
            self.global_num += 1

        return {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time}
