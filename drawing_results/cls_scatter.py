import matplotlib.pyplot as plt
import numpy as np


Labels = ['ShuffleNetv2 (2.0x)', 'MobileNetv3 (1.0x)',  'EfficientNet-B0', 'ConvNext-T (0.6x)', 'PVT-T', 'Swin-1G', 'DeIT-2G', 'ConViT-T', 'LeViT-128S', 'Mobile-Former', 'MobileViT-S(baseline)', 'EdgeFormer-S (Ours)']
# the last two elements in each vector are manually set for normalizing
Params = [5.5,  5.4  , 5.3  , 10 , 13.2, 7.3  , 9.5  , 6.0  , 7.8  , 9.4  , 5.6  , 5.0]
Top1 =   [74.5, 75.2 , 76.3 , 77.9, 75.1, 77.3 , 77.6 , 73.1 , 76.6 , 76.7 , 78.4 , 78.6]
# normalization
Params=np.array(Params)
Top1 = np.array(Top1)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# # using heat map to show model size
# plt.scatter(Params, Top1, c=Params, s=Params*20, alpha=0.3, cmap='viridis', marker='d')
# plt.ylim((74.0, 80.0))
# plt.xlim((4, 14))
# plt.ylabel('Top1 (%)')
# plt.xlabel('# params. (M)')
# plt.colorbar()
# plt.savefig('./result/cls.png')
# plt.show()

# using heat map to show model size
plt.scatter(Params, Top1, s=100, alpha=0.3, cmap='viridis', marker='d')
plt.ylim((74.0, 80.0))
plt.xlim((4, 14))
plt.ylabel('Top1 (%)')
plt.xlabel('# params. (M)')
for x, y, label in zip(Params, Top1, Labels):
    plt.annotate(label, xy=(x-0.1,y+0.15))

plt.savefig('./result/cls.png')
plt.show()


print('done')
