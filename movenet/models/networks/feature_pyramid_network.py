'''
Based on torchvision.ops.feature_pyramid_network.
In the original paper, they `fix the feature dimension (numbers of channels, denoted as d) in all the feature maps.`
However, by diving into the Movenet, I found out that the feature dimension is incrementally decreased, from 64 to 32 to 24. So I made the changes correspondingly.
'''

from collections import OrderedDict

import torch.nn.functional as F
from torch import nn, Tensor

from typing import Tuple, List, Dict, Optional

class SeperableConv(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        activation_layer = None
    ) -> None:
        super(SeperableConv, self).__init__()

        if activation_layer is None:
            activation_layer = nn.ReLU

        hidden_dim = int(round(inp))

        layers: List[nn.Module] = []
        layers.extend([
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=True),
            activation_layer(inplace=True),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """
    def __init__(
        self,
        in_channels_list: List[int], # [24, 32, 64, 1280]
        out_channels_list: List[int], # [24, 32, 64, 64]
        fused_channels_list = List[int], # [24, 24, 32]
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        assert len(in_channels_list) == len(out_channels_list), 'The lengths of in_channels_list and out_channels_list should be equal.'
        for i in range(len(in_channels_list)):
            in_channels = in_channels_list[i]
            out_channels = out_channels_list[i]
            if in_channels == 0 or out_channels == 0:
                raise ValueError("in_channels=0/out_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            self.inner_blocks.append(inner_block_module)
            if i != len(in_channels_list) - 1:
                fused_channels = fused_channels_list[i]
                layer_block_module = SeperableConv(out_channels, fused_channels)
                self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (Tensor): highest  maps after FPN layers.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)


        for idx in range(len(x)-2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)

            # for pytorch inference
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="bilinear", align_corners=False)
            # for model convertion, please comment the above line and uncomment the following line.
            # inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            last_inner = self.get_result_from_layer_blocks(last_inner, idx)
        
        return last_inner
