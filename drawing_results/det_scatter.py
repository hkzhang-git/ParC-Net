import matplotlib.pyplot as plt
import numpy as np


Labels = ['MobileNetV1', 'MNASNet', 'MobileViT-XS', 'ResNet50', 'MobileViT-S (Baseline)', 'EdgeFormer-S (Ours)']
# the last two elements in each vector are manually set for normalizing
Params = [5.1,  4.9, 2.7, 22.9, 5.7,  5.2]
mAP =    [22.2, 23.0, 24.8, 25.2, 27.7, 28.8]
# normalization
Params=np.array(Params)
mAP = np.array(mAP)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# # using heat map to show model size
# plt.scatter(Params, mAP, c=Params, s=Params*20, alpha=0.3, cmap='viridis', marker='d')
# plt.ylim((21.0, 30.0))
# plt.xlim((2.0, 24.0))
# plt.ylabel('mAP')
# plt.xlabel('# params. (M)')
# plt.colorbar()
# plt.savefig('./result/det.png')
# plt.show()

plt.scatter(Params, mAP, s=100, alpha=0.3, cmap='viridis', marker='d')
plt.ylim((21.0, 30.0))
plt.xlim((2.0, 26.0))
plt.ylabel('mAP')
plt.xlabel('# params. (M)')
for x, y, label in zip(Params, mAP, Labels):
    plt.annotate(label, xy=(x-0.1, y+0.22))
plt.savefig('./result/det.png')
plt.show()

print('done')
