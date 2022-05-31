from __future__ import print_function
import os
import sys
import numpy as np
from PIL import Image


def main():
    ext = '.png'
    path = '/home/disk/data/pascal_aug/VOC2012/SegmentationClass'
    txt_file = '/home/disk/data/pascal_aug/VOC2012/ImageSets/Segmentation/trainval.txt'
    path_converted = '/home/disk/data/pascal_aug/VOC2012/SegmentationClass_1D'

    # Create dir for converted labels
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    with open(txt_file, 'r') as f:
        for img_name in f:
            img_base_name = img_name.strip()
            print(img_base_name)
            img_name = os.path.join(path, img_base_name) + ext

            mask = Image.open(img_name)
            mask = np.array(mask).astype('int32')
            mask = Image.fromarray(mask.astype('uint8'))
            mask.save(os.path.join(path_converted, img_base_name + ext))


def process_arguments(argv):
    if len(argv) != 4:
        help()

    path = argv[1]
    list_file = argv[2]
    new_path = argv[3]

    return path, list_file, new_path


def help():
    print('Usage: python convert_labels.py PATH LIST_FILE NEW_PATH\n'
          'PATH points to directory with segmentation image labels.\n'
          'LIST_FILE denotes text file containing names of images in PATH.\n'
          'Names do not include extension of images.\n'
          'NEW_PATH points to directory where converted labels will be stored.'
          , file=sys.stderr)
    exit()


if __name__ == '__main__':
    main()