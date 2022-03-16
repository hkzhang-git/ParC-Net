
# EdgeFormer: Improving Light-weight ConvNets by Learning from Vision Transformers

Official PyTorch implementation of **EdgeFormer**

---
<p align="center">
<img src="https://github.com/hkzhang91/EdgeFormer/tree/main/drawing_results/result/results.png" width=100% height=100% 
class="center">
</p>

## Results and Pre-trained Models

### EdgeFormer-S
| Tasks | performance | #params | pretrained models |
|:---:|:---:|:---:|:---:|
| Classification | 78.6 (Top1 acc) | 5.0 | [model](https://github.com/hkzhang91/EdgeFormer/blob/main/pretrained_models/classification/checkpoint_ema_avg.pt) |
| Detection      | 28.8 (mAP)      | 5.2 | [model](https://github.com/hkzhang91/EdgeFormer/blob/main/pretrained_models/detection/checkpoint_ema_avg.pt) |
| Segmentation   | 79.7 (mIOU)     | 5.8 | [model](https://github.com/hkzhang91/EdgeFormer/blob/main/pretrained_models/segmentation/checkpoint_ema_avg.pt) |
