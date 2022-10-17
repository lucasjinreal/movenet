import warnings
import torch
from torch import nn
from .feature_pyramid_network import FeaturePyramidNetwork


from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models import mobilenet
from .mobilenetv2 import mobilenet_v2


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()


        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels_list=[24, 32, 64, 64],
            fused_channels_list=[24, 24, 32],
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def mobilenet_backbone(
    backbone_name, # discared as we always use mobilenet v2
    pretrained,
    fpn,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=2,
    returned_layers=None,
    extra_blocks=None,
    model_type='lighting'
):
    if model_type == 'lighting':
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
    else:
        inverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],
            [6, 40, 2, 2],
            [6, 56, 3, 2],
            [6, 112, 4, 2],
            [6, 168, 3, 1],
            [6, 280, 3, 2],
            [6, 560, 1, 1],
        ]

    backbone = mobilenet_v2(pretrained=pretrained, norm_layer=norm_layer, inverted_residual_setting = inverted_residual_setting).features
    # print("backbone: ", backbone)

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    # mli: for mobilenet, the obtained stage_indices = [0, 2, 4, 7, 14, 18]
    # stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    # mli: the following block indices refer to the last layer of each stage (s4, s8, s16, s32)
    # **This is wrong** stage_indices = [2, 4, 7, 14]
    stage_indices = [3, 6, 10, 18]
    num_stages = len(stage_indices)
    # print("# stages: ", num_stages)
    # print("Stage indicse: ", stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    # mli: make all layers trainable.
    # for b in backbone[:freeze_before]:
    #     for parameter in b.parameters():
    #         parameter.requires_grad_(False)

    out_channels = 24
    if fpn:
        # mli: remove the extra_blocks
        # if extra_blocks is None:
        #     extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = list(range(num_stages))
        assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
        return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}
        # print("Return layers: ", return_layers)

        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        # print("in_channels_list", in_channels_list)

        return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    else:
        m = nn.Sequential(
            backbone,
            # depthwise linear combination of channels to reduce their size
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels
        return m

'''
# test the functionality
if __name__=='__main__':
    """
    Constructs a specified MobileNet v2 backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]
    """
    backbone = mobilenet_backbone('mobilenet_v2', fpn=True, pretrained=False, trainable_layers=3)
    x = torch.rand(1,3,192,192)
    # compute the output
    output = backbone(x)
    print('output shape: ', output.shape)
'''
