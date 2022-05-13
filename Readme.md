
# EdgeFormer: Improving Light-weight ConvNets by Learning from Vision Transformers

[中文版](https://github.com/hkzhang91/EdgeFormer/blob/main/REDAME_ch.md)

Official PyTorch implementation of **EdgeFormer**

---
<p align="center">
<img src="https://s1.ax1x.com/2022/03/16/qpPfb9.png" width=100% height=100% 
class="center">
</p>

<p align="center">
<img src="https://s1.ax1x.com/2022/03/16/qpE27Q.png" width=60% height=60% 
class="center">
</p>


## Introduction
EdgeFormer, a pure ConvNet based light weight backbone model that inherits advantages of ConvNets and integrates strengths of vision transformers. Specifically, we propose global circular convolution (GCC) with position embeddings, a light-weight convolution op which boasts a global receptive field while producing location sensitive features as in local convolutions. We combine the GCCs and squeeze-exictation ops to form a meta-former like model block, which further has the attention mechanism like transformers. The aforementioned block can be used in plug-and-play manner to replace relevant blocks in ConvNets or transformers. Experiment results show that the proposed EdgeFormer achieves better performance than popular light-weight ConvNets and vision transformer based models in common vision tasks and datasets, while having fewer parameters and faster inference speed. For classification on ImageNet-1k, EdgeFormer achieves 78.6% top-1 accuracy with about 5.0 million parameters, saving 11% parameters and 13% computational cost but gaining 0.2% higher accuracy and 23% faster inference speed (on ARM based Rockchip RK3288) compared with MobileViT. 

## EdgeFormer block
<p align="center">
<img src="https://s1.ax1x.com/2022/03/16/qpaZwQ.png" width=60% height=60% 
class="center">
</p>

## Global circular convolution

<p align="center">
<img src="https://s1.ax1x.com/2022/03/16/qpaeoj.png" width=60% height=60% 
class="center">
</p>


## Experimental results

### EdgeFormer-S
| Tasks | performance | #params | pretrained models |
|:---:|:---:|:---:|:---:|
| Classification | 78.6 (Top1 acc) | 5.0 | [model](https://github.com/hkzhang91/EdgeFormer/blob/main/pretrained_models/classification/checkpoint_ema_avg.pt) |
| Detection      | 28.8 (mAP)      | 5.2 | [model](https://github.com/hkzhang91/EdgeFormer/blob/main/pretrained_models/detection/checkpoint_ema_avg.pt) |
| Segmentation   | 79.7 (mIOU)     | 5.8 | [model](https://github.com/hkzhang91/EdgeFormer/blob/main/pretrained_models/segmentation/checkpoint_ema_avg.pt) |

### Inference speed
We deploy the proposed EdgeFormer and baseline on widely used low power chip Rockchip RK3288 and our own micro-power chip for comparison. We use ONNX [1] and MNN to port these models to RK3288 and micro-power chip and time each model for 100 iterations to measure the average inference speed.

| Models | #params (M) | Madds (M)| RK3288 inference speed (ms) | micro-power chip (ms)| Top1 acc |
|:---:|:---:|:---:|:---:|:---:|:---:|
| MobileViT-S | 5.6 | 2010 |  457| 368 | 78.4 |
| EdgeFormer-S | 5.0 (-11%)| 1740 (-13%) | 353 (+23%)| 98 (3.77x) | 78.6 (+0.2%) |

### Combination of EdgeFormer and ConvNext

Classification experiments
| Models | # params | Top1 acc |
|:---:|:---:|:---:|
| ConvNext-XT       | 7.44 (M) | 77.5 |
| ConvNext-GCC-XT   | 7.41 (M) | 78.3 (+0.8)|
| ConvNext-T        |          | training |
| ConvNext-GCC-T    |          | training |

Detection experiments
| Models | # params | AP box  |  AP50 box  |  AP75 box  |  AP mask   |  AP50 mask  |  AP75 mask |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNext-XT       | - | 47.2  |  65.6   |  51.4  |  41.0  |  63.0  |  44.2 |
| ConvNext-GCC-XT   | - | 47.7  |  66.2   |  52.0  |  41.5  |  63.6  |  44.6 |
| ConvNext-T        | - | training |      |        |        |        |       |
| ConvNext-GCC-T    | - | training |      |        |        |        |       |


<p align="center">
<img src="https://s1.ax1x.com/2022/04/28/LOoPu4.png" width=40% height=40% 
class="center">
</p>

In terms of designing a pure ConvNet via learning from ViTs, our proposed EdgeFormer is most closely related to a parallel work ConvNext. By comparing Edgeformer with Convnext, we notice that their improvements are different and complementary. To verify this point, we build a combination network, where Edgeformer blocks are used to replace several ConvNext blocks in the end of last two stages. Experiment results show that **the replacement signifcantly improves classification accuracy, while slightly decreases the number of parameters**. Corresponding code will be released soon. 





## Installation
We implement the EdgeFomer with PyTorch-1.9.0, CUDA=11.1. 
### PiP
The environment can be build in the local python environment using the below command:
``` 
pip install -r requirements.txt
```
### Dokcer
A docker image containing the environment will be provided soon. 

## Training
Training settings are listed in yaml files (./config/classification/xxx/xxxx.yaml, ./config/detection/xxx/xxxx.yaml, ./config/segmentation/xxx/xxxx.yaml  )

Classifiction
``` 
cd EdgeFormer-main
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_train.py --common.config-file ./config/classification/edgeformer/edgeformer_s.yaml
``` 
Detection
``` 
cd EdgeFormer-main
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train.py --common.config-file --common.config-file config/detection/ssd_edgeformer_s.yaml
``` 
Segmentation
``` 
cd EdgeFormer-main
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_train.py --common.config-file --common.config-file config/segmentation/deeplabv3_edgeformer_s.yaml
``` 

## Evaluation

Classifiction
``` 
cd EdgeFormer-main
CUDA_VISIBLE_DEVICES=0 python eval_cls.py --common.config-file ./config/classification/edgeformer/edgeformer_s.yaml --model.classification.pretrained ./pretrained_models/classification/checkpoint_ema_avg.pt
``` 

Detection
``` 
cd EdgeFormer-main
CUDA_VISIBLE_DEVICES=0 python eval_det.py --common.config-file ./config/detection/edgeformer/ssd_edgeformer_s.yaml --model.detection.pretrained ./pretrained_models/detection/checkpoint_ema_avg.pt --evaluation.detection.mode validation_set --evaluation.detection.resize-input-images
``` 

Segmentation
``` 
cd EdgeFormer-main
CUDA_VISIBLE_DEVICES=0 python eval_seg.py --common.config-file ./config/detection/edgeformer/deeplabv3_edgeformer_s.yaml --model.segmentation.pretrained ./pretrained_models/segmentation/checkpoint_ema_avg.pt --evaluation.segmentation.mode validation_set --evaluation.segmentation.resize-input-images
``` 

## Acknowledgement
We thank authors of MobileVit for sharing their code. We implement our EdgeFormer based on their [source code](https://github.com/apple/ml-cvnets). If you find this code is helpful in your research, please consider citing our paper and [MobileVit](https://arxiv.org/abs/2110.02178?context=cs.LG)




