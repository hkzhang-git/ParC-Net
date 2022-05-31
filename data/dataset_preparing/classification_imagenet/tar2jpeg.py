import os
import subprocess
from glob import glob
from functools import partial
from multiprocessing import Pool


data_root = '/home/disk/data/imagenet1k/train'
img_root = '/home/disk/data/imagenet1k/training'

tar_list = glob(data_root + '/*.tar')



def untar(id, tar_list):
    tar=tar_list[id]
    tar_name = tar.split('/')[-1].split('.')[0]
    jpeg_dir = img_root + '/{}'.format(tar_name)
    os.makedirs(jpeg_dir)
    cmd = 'tar xf {} -C {}'.format(tar, jpeg_dir)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    pool = Pool()
    partial_untar = partial(untar, tar_list=tar_list)
    N = len(tar_list)
    _ = pool.map(partial_untar, range(N))
    pool.close()
    pool.join()







