from ParC_ConvNets.ParC_convnext import parc_convnext_xt
from ParC_ConvNets.ParC_resnet50 import parc_res50
from ParC_ConvNets.ParC_mobilenetv2 import parc_mv2
import torch

input=torch.randn(2,3,224,224)
model = parc_convnext_xt()
out=model(input)
print('done')
