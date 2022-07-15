
# PaCC-Net: Position Aware Circular Convolution with Merits from ConvNets and Transformer

[中文版](https://github.com/hkzhang91/EdgeFormer/blob/main/REDAME_ch.md)

[PaCC-Net](https://arxiv.org/abs/2203.03952) ECCV 2022

This reposity was named EdgeFormer, which is changed to PaCC-Net, as "Former" indicates that the model is some variant of transformer.

Official PyTorch implementation of **PaCC-Net**

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
Recently, vision transformers started to show impressive results which outperform large convolution based models significantly. However, in the area of small models for mobile or resource constrained devices, ConvNet still has its own advantages in both performance and model complexity. We propose PaCC-Net, a pure ConvNet based backbone model that further strengthens these advantages by fusing the merits of vision transformers into ConvNets.  Specifically, we propose position aware circular convolution (PaCC), a light-weight convolution op which boasts a global receptive field while producing location sensitive features as in local convolutions. We combine the PaCCs and squeeze-exictation ops to form a meta-former like model block, which further has the attention mechanism like transformers. The aforementioned block can be used in plug-and-play manner to replace relevant blocks in ConvNets or transformers. Experiment results show that the proposed PaCC-Net achieves better performance than popular light-weight ConvNets and vision transformer based models in common vision tasks and datasets, while having fewer parameters and faster inference speed. For classification on ImageNet-1k, PaCC-Net achieves 78.6% top-1 accuracy with about 5.0 million parameters, saving 11% parameters and 13% computational cost but gaining 0.2% higher accuracy and 23% faster inference speed (on ARM based Rockchip RK3288) compared with MobileViT, and uses only 0.5× parameters but gaining 2.7% accuracy compared with DeIT. On MS-COCO object detection and PASCAL VOC segmentation tasks, PaCC-Net also shows better performance.

## PaCC block
<p align="center">
<img src="https://s1.ax1x.com/2022/03/16/qpaZwQ.png" width=60% height=60% 
class="center">
</p>

## Position aware circular convolution

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
We deploy the proposed EdgeFormer and baseline on widely used low power chip Rockchip RK3288 and DP2000 chip for comparison. DP2000 is the code name of
a in house unpublished low power neural network processor that highly optimizes the convolutions. We use ONNX [1] and MNN to port these models to RK3288 and DP2000 chip and time each model for 100 iterations to measure the average inference speed.

| Models | #params (M) | Madds (M)| RK3288 inference speed (ms) | DP2000 (ms)| Top1 acc |
|:---:|:---:|:---:|:---:|:---:|:---:|
| MobileViT-S | 5.6 | 2010 |  457| 368 | 78.4 |
| PaCC-Net-S | 5.0 (-11%)| 1740 (-13%) | 353 (+23%)| 98 (3.77x) | 78.6 (+0.2%) |

### Applying Edgeformer designs on various lightweight backbones
Classification experiments. CPU used here is Xeon E5-2680 v4. *Authors of EdgeViT do not clarify the type of CPU used in their paper.

| Models         |# params |Madds   |Devices |Speed(ms) |Top1 acc| Source |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MobileViT-S     | 5.6 M   | 2.0G   | RK3288 | 457 | 78.4 | ICLR 22 |   
|PaCC-Net-S    | 5.0 M   | 1.7G   | RK3288 | 353 | 78.6 | Ours    |
|MobileViT-S     | 5.6 M   | 2.0G   | DP2000 | 368 | 78.4 | ICLR 22 |
|PaCC-Net-S    | 5.0 M   | 1.7G   | DP2000 | 98  | 78.6 | Ours    |
|ResNet50        | 26 M    | 2.1G   | CPU    | 98  | 78.8 | CVPR 16 |
|PaCC-ResNet50    | 24 M    | 2.0G   | CPU    | 98  | 79.6 | Ours    |
|MobileNetV2     | 3.5 M   | 0.3G   | CPU    | 24  | 70.2 | CVPR 18 |
|PaCC-MobileNetV2 | 3.5 M   | 0.3G   | CPU    | 27  | 71.1 | Ours    |
|ConvNext-XT     | 7.4 M   | 0.6G   | CPU    | 47  | 77.5 | CVPR 22 |
|PaCC-ConvNext-XT | 7.4 M   | 0.6G   | CPU    | 48  | 78.3 | Ours    |
|EdgeViT-XS      | 6.7 M   | 1.1G   | CPU*   | 54* | 77.5 | Arxiv 22/05 |
|PaCC-EdgeViT-XS  |         |        |        |     |      |         |


Detection experiments
| Models | # params | AP box  |  AP50 box  |  AP75 box  |  AP mask   |  AP50 mask  |  AP75 mask |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNext-XT       | - | 47.2  |  65.6   |  51.4  |  41.0  |  63.0  |  44.2 |
| PaCC-ConvNext-XT   | - | 47.7  |  66.2   |  52.0  |  41.5  |  63.6  |  44.6 |


Segmentation experiments

To be done

ConvNext block and ConvNext-GCC block
<p align="center">
<img src="https://s1.ax1x.com/2022/05/16/OWxqNd.png" width=40% height=40% 
class="center">
</p>

In terms of designing a pure ConvNet via learning from ViTs, our proposed PaCC-Net is most closely related to a parallel work ConvNext. By comparing PaCC-Net with Convnext, we notice that their improvements are different and complementary. To verify this point, we build a combination network, where PaCC blocks are used to replace several ConvNext blocks in the end of last two stages. Experiment results show that **the replacement signifcantly improves classification accuracy, while slightly decreases the number of parameters**. Results on ResNet50, MobileNetV2 and ConvNext-T shows that models which focus on optimizing FLOPs-accuracy trade-offs can also benefit from our PaCC-Net designs. Corresponding code will be released soon. 

## Installation
We implement the PaCC-Net with PyTorch-1.9.0, CUDA=11.1. 
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


```
@inproceedings{zhang2022edgeformer,
  title={EdgeFormer: Improving Light-weight ConvNets by Learning from Vision Transformers},
  author={Zhang, Haokui and Hu, Wenze and Wang, Xiaoyu},
  booktitle={European Conference on Computer Vision},
  pages={},
  year={2022}
}
```
```
@inproceedings{mehta2021mobilevit,
  title={Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={ICLR},
  year={2022}
}
```


