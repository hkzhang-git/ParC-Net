from typing import Dict

from utils import logger


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.classification.edge.mode", "outer_frame_v1")
    if mode is None:
        logger.error("Please specify mode")

    mode = mode.lower()
    if mode in ['outer_frame_v1', 'outer_frame_v2']:
        scale = getattr(opts, "model.classification.edge.scale", 'scale_s')
        kernel = getattr(opts, "model.classification.edge.kernel", "gcc_ca")
        fusion = getattr(opts, "model.classification.edge.fusion", "concat")
        instance_kernel = getattr(opts, "model.classification.edge.instance_kernel", "interpolation_bilinear")
        mid_mix = getattr(opts, "model.classification.edge.mid_mix", False)
        use_pe = getattr(opts, "model.classification.edge.use_pe", True)

        # the sizes of output of layer3-5 are 32, 16 and 8
        if 'big_kernel_1-2' in kernel:
            big_kernel_sizes = [17, 9, 5]
        elif 'big_kernel_1-4' in kernel:
            big_kernel_sizes = [9, 5, 3]
        else:
            big_kernel_sizes = [0, 0, 0]

        if scale == 'scale_h':
            config = {
                "layer1": {
                    "out_channels": 96,
                    "expand_ratio": 4,
                    "num_blocks": 1,
                    "stride": 1,
                    "block_type": "mv2"
                },
                "layer2": {
                    "out_channels": 144,
                    "expand_ratio": 4,
                    "num_blocks": 3,
                    "stride": 2,
                    "block_type": "mv2"
                },
                "layer3": {  # 20-40, 256/8=32
                    "out_channels": 216,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 324,
                    "cf_ffn_channels":648 ,
                    "cf_blocks": 3,
                    "big_kernel_size": big_kernel_sizes[0],
                    "meta_kernel_size": 32,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "layer4": {  # 10-20, 256/16=16
                    "out_channels": 288,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 432,
                    "cf_ffn_channels":864,
                    "cf_blocks": 9,
                    "big_kernel_size": big_kernel_sizes[1],
                    "meta_kernel_size": 16,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "layer5": {  # 5-10, 256/32=8
                    "out_channels": 432,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 432,
                    "cf_ffn_channels": 864,
                    "cf_blocks": 3,
                    "big_kernel_size": big_kernel_sizes[2],
                    "meta_kernel_size": 8,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "last_layer_exp_factor": 4
            }
        elif scale == 'scale_s':
            config = {
                "layer1": {
                    "out_channels": 32,
                    "expand_ratio": 4,
                    "num_blocks": 1,
                    "stride": 1,
                    "block_type": "mv2"
                },
                "layer2": {
                    "out_channels": 64,
                    "expand_ratio": 4,
                    "num_blocks": 3,
                    "stride": 2,
                    "block_type": "mv2"
                },
                "layer3": {  # 20-40, 256/8=32
                    "out_channels": 96,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 192,
                    "cf_ffn_channels": 384,
                    "cf_blocks": 2,
                    "big_kernel_size": big_kernel_sizes[0],
                    "meta_kernel_size": 32,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "layer4": {  # 10-20, 256/16=16
                    "out_channels": 128,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 256,
                    "cf_ffn_channels": 512,
                    "cf_blocks": 4,
                    "big_kernel_size": big_kernel_sizes[1],
                    "meta_kernel_size": 16,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "layer5": {  # 5-10, 256/32=8
                    "out_channels": 160,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 320,
                    "cf_ffn_channels": 640,
                    "cf_blocks": 3,
                    "big_kernel_size": big_kernel_sizes[2],
                    "meta_kernel_size": 8,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "last_layer_exp_factor": 4
            }
        elif scale == 'scale_xs':
            config = {
                "layer1": {
                    "out_channels": 32,
                    "expand_ratio": 4,
                    "num_blocks": 1,
                    "stride": 1,
                    "block_type": "mv2"
                },
                "layer2": {
                    "out_channels": 64,
                    "expand_ratio": 4,
                    "num_blocks": 3,
                    "stride": 2,
                    "block_type": "mv2"
                },
                "layer3": {  # 20-40, 256/8=32
                    "out_channels": 96,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 96,
                    "cf_ffn_channels": 192,
                    "cf_blocks": 2,
                    "big_kernel_size": big_kernel_sizes[0],
                    "meta_kernel_size": 32,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "layer4": {  # 10-20, 256/16=16
                    "out_channels": 128,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 128,
                    "cf_ffn_channels": 256,
                    "cf_blocks": 3,
                    "big_kernel_size": big_kernel_sizes[1],
                    "meta_kernel_size": 16,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "layer5": {  # 5-10, 256/32=8
                    "out_channels": 160,
                    "mv_expand_ratio": 4,
                    "stride": 2,
                    "block_type": mode,
                    "kernel": kernel,
                    "fusion": fusion,
                    "cf_s_channels": 160,
                    "cf_ffn_channels": 320,
                    "cf_blocks": 2,
                    "big_kernel_size": big_kernel_sizes[2],
                    "meta_kernel_size": 8,
                    "instance_kernel_method": instance_kernel,
                    "use_pe": use_pe,
                    "mid_mix": False,
                    "ffn_dropout": 0.0,
                    "dropout": 0.1
                },
                "last_layer_exp_factor": 4
            }

    else:
        raise NotImplementedError

    return config