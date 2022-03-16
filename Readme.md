
# EdgeFormer: Improving Light-weight ConvNets by Learning from Vision Transformers

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
EdgeFormer, a pure ConvNet based light weight backbone model that inherits advantages of ConvNets and integrates strengths of vision transformers. Specifically, we propose global
circular convolution (GCC) with position embeddings, a light-weight convolution op which boasts a global receptive field while producing location sensitive features as in local convolutions. We combine the GCCs and squeeze-exictation ops to form a meta-former like model block, which further has the attention mechanism like transformers. The aforementioned block can be used in plug-and-play manner to replace relevant blocks in ConvNets or transformers. Experiment results show that the proposed EdgeFormer achieves better performance than popular light-weight ConvNets and vision transformer based models in common vision tasks and datasets, while having fewer parameters and faster inference
speed. For classification on ImageNet-1k, EdgeFormer achieves 78.6% top-1 accuracy with about 5.0 million parameters, saving 11% parameters and 13% computational cost but gaining 0.2% higher accuracy and 23% faster inference speed (on ARM based Rockchip RK3288) compared with MobileViT. 

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


## Installation
We implement EdgeFomer with PyTorch-1.9.0, CUDA=11.1. 
### PiP
The environment can be build in the local python environment using the below command:
``` 
pip install -r requirements.txt
```
### Dokcer
A docker image containing environment will be provided soon. 

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
We thanks authors of Mobilevit for sharing their code. We implement EdgeFormer based on their [source code](https://github.com/apple/ml-cvnets). If you find this code useful, please cite our paper and [Mobilevit](https://arxiv.org/abs/2110.02178?context=cs.LG)




