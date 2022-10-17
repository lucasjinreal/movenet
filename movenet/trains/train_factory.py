from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .multi_pose import MultiPoseTrainer
from .single_pose import SinglePoseTrainer

train_factory = {
  'multi_pose': MultiPoseTrainer,
  'single_pose': SinglePoseTrainer
}
