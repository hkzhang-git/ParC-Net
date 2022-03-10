import matplotlib.pyplot as plt
import numpy as np


Labels = ['MobileNetV1', 'MobileNetV2',  'MobileViT-S' , 'MobileViT-S (baseline)', 'EdgeFormer-S (Ours)']
Params = [11.2, 4.5, 2.9, 6.4, 5.8]
mIOU = [75.3, 75.7, 77.1, 79.1, 79.7]

Params=np.array(Params)
mIOU = np.array(mIOU)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# # using heat map to show model size
# plt.scatter(Params, mIOU, c=Params, s=Params*20, alpha=0.3, cmap='viridis', marker='d')
# plt.ylim((75.0, 80.5))
# plt.ylabel('mIOU')
# plt.xlabel('# params. (M)')
# plt.colorbar()
# plt.savefig('./result/seg.png')
# plt.show()

plt.scatter(Params, mIOU, s=100, alpha=0.3, cmap='viridis', marker='d')
plt.ylim((75.0, 80.5))
plt.xlim(2.5, 13.0)
plt.ylabel('mIOU')
plt.xlabel('# params. (M)')
for x, y, label in zip(Params, mIOU, Labels):
    plt.annotate(label, xy=(x-0.1,y+0.15))
plt.savefig('./result/seg.png')
plt.show()

print('done')
