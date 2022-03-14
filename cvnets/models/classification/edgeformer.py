
from torch import nn
import argparse
from typing import Dict, Tuple, Optional

from utils import logger

from . import register_cls_models
from .base_cls import BaseEncoder
from .config.edgeformer import get_configuration
from ...layers import ConvLayer, LinearLayer, GlobalPool, Dropout, SeparableConv
from ...modules import InvertedResidual, outer_frame_v1, outer_frame_v2


@register_cls_models("edgeformer")
class edgeformer(BaseEncoder):
    """
        our proposed model
    """
    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        # for edgeformer_h
        scale = getattr(opts, "model.classification.edge.scale", 'scale_s')

        # chip_former_config
        cf_config = get_configuration(opts=opts)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(edgeformer, self).__init__()
        self.dilation = 1

        # store model configuration in a dictionary
        self.model_conf_dict = dict()

        if 'scale_h' in scale:
            self.conv_1 = ConvLayer(
                opts=opts, in_channels=image_channels, out_channels=48,
                kernel_size=5, stride=2, use_norm=True, use_act=True
            )
            out_channels=48
        else:
            self.conv_1 = ConvLayer(
                opts=opts, in_channels=image_channels, out_channels=out_channels,
                kernel_size=3, stride=2, use_norm=True, use_act=True
            )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=cf_config["layer1"]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=cf_config["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=cf_config["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=cf_config["layer4"], dilate=dilate_l4
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=cf_config["layer5"], dilate=dilate_l5
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        exp_channels = min(cf_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
                opts=opts, in_channels=in_channels, out_channels=exp_channels,
                kernel_size=1, stride=1, use_act=True, use_norm=True
            )

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--model.classification.edge.mode', type=str, default=None,
                           choices=['outer_frame_v1', 'outer_frame_v2'], help="outer frame")
        group.add_argument('--model.classification.edge.scale', type=str, default='scale_s',
                           choices=['scale_xs', 'scale_s', 'scale_h'], help="model scale")
        group.add_argument('--model.classification.edge.kernel', type=str, default="gcc_ca",
                           choices=['gcc_ca', 'gcc', 'bkc_ca', 'bkc'])
        group.add_argument('--model.classification.edge.fusion', type=str, default="concat",
                           choices=['add', 'concat'])
        group.add_argument('--model.classification.edge.instance_kernel', type=str, default="crop",
                           choices=['crop', 'interpolation_bilinear'])
        group.add_argument('--model.classification.edge.mid_mix', type=bool, default=False)
        group.add_argument('--model.classification.edge.use_pe', type=bool, default=False)

        return parser

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "edgeformer")
        if "outer_frame" in block_type.lower():
            if block_type.lower() == "outer_frame_v1":
                return self._make_outer_frame_v1(
                    opts=opts,
                    input_channel=input_channel,
                    cfg=cfg,
                    dilate=dilate
                )
            elif block_type.lower() == "outer_frame_v2":
                return self._make_outer_frame_v2(
                    opts=opts,
                    input_channel=input_channel,
                    cfg=cfg,
                    dilate=dilate
                )
        else:
            return self._make_mb_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mb_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_outer_frame_v1(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            self.residual = InvertedResidual(opts=opts, in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                             stride=stride, expand_ratio=cfg.get("mv_expand_ratio", 4),
                                             dilation=prev_dilation)

            layer = self.residual
            input_channel = cfg.get("out_channels")

            block.append(layer)

        block.append(
            outer_frame_v1(
                opts=opts,
                meta_encoder=cfg.get("kernel"),
                in_channels=cfg.get("out_channels"),
                cf_s_channels=cfg.get("cf_s_channels"),
                n_blocks=cfg.get("cf_blocks"),
                meta_kernel_size=cfg.get("meta_kernel_size"),
                big_kernel_size=cfg.get("big_kernel_size"),
                instance_kernel_method=cfg.get("instance_kernel_method"),
                fusion_method=cfg.get("fusion"),
                use_pe=cfg.get("use_pe"),
                mid_mix=cfg.get("mid_mix"),
                bias=cfg.get("bias"),
                cf_ffn_channels=cfg.get("cf_ffn_channels"),
                ffn_dropout=cfg.get("ffn_dropout"),
                dropout=cfg.get("dropout"),
            )
        )

        return nn.Sequential(*block), input_channel

    def _make_outer_frame_v2(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            self.residual = nn.Sequential(
                nn.BatchNorm2d(input_channel),
                nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1),
                InvertedResidual(opts=opts, in_channels=input_channel, out_channels=cfg.get("out_channels"),
                                 stride=1, expand_ratio=cfg.get("mv_expand_ratio", 4),
                                 dilation=prev_dilation)
            )

            layer = self.residual
            input_channel = cfg.get("out_channels")

            block.append(layer)

        block.append(
            outer_frame_v2(
                opts=opts,
                meta_encoder=cfg.get("kernel"),
                in_channels=cfg.get("out_channels"),
                cf_s_channels=cfg.get("cf_s_channels"),
                n_blocks=cfg.get("cf_blocks"),
                meta_kernel_size=cfg.get("meta_kernel_size"),
                big_kernel_size=cfg.get("big_kernel_size"),
                instance_kernel_method=cfg.get("instance_kernel_method"),
                fusion_method=cfg.get("fusion"),
                use_pe=cfg.get("use_pe"),
                mid_mix=cfg.get("mid_mix"),
                bias=cfg.get("bias"),
                cf_ffn_channels=cfg.get("cf_ffn_channels"),
                ffn_dropout=cfg.get("ffn_dropout"),
                dropout=cfg.get("dropout"),
            )
        )

        return nn.Sequential(*block), input_channel



