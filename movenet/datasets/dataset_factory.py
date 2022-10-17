from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .sample.multi_pose import MultiPoseDataset
from .sample.single_pose import SinglePoseDataset

from .dataset.coco_hp import COCOHP
from .dataset.active import ACTIVE


dataset_factory = {
  'coco_hp': COCOHP,
  'active': ACTIVE,
  'active_coco': ACTIVE,
}

_sample_factory = {
  'multi_pose': MultiPoseDataset,
  'single_pose': SinglePoseDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
