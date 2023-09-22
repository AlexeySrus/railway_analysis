from argparse import ArgumentParser, Namespace
import numpy as np
import os
import json
from tqdm import tqdm

CLASSES_NAMES = ['Car', 'Human', 'Wagon', 'FacingSwitchL', 'FacingSwitchNV', 'TrailingSwitchL', 'TrailingSwitchNV', 'SignalE', 'SignalF']


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Convert RZD dataset to YOLO format')
    parser.add_argument('--images', type=str, required=True, help='Path to folder with images split to train/val/test subfolders')
    parser.add_argument('--boxes', type=str, required=True, help='Path to folder with rzd labels')
    parser.add_argument('--labels', type=str, required=True, help='Path to result yolo labels folde')
    return parser.parse_args()


def get_class_id(category: str) -> int:
    if category == 'FacingSwitchR':
        category = 'FacingSwitchL'
    if category == 'TrailingSwitchR':
        category = 'TrailingSwitchL'
    return CLASSES_NAMES.index(category)


if __name__ == '__main__':
    args = parse_args()

    train_images_folder = os.path.join(args.images, 'train/')
    val_images_folder = os.path.join(args.images, 'val/')
    test_images_folder = os.path.join(args.images, 'test/')

    val_images_basenames = [os.path.splitext(p)[0] for p in os.listdir(val_images_folder)]
    test_images_basenames = [os.path.splitext(p)[0] for p in os.listdir(test_images_folder)]

    train_labels_folder = os.path.join(args.labels, 'train/')
    val_labels_folder = os.path.join(args.labels, 'val/')
    test_labels_folder = os.path.join(args.labels, 'test/')

    for folder in [train_labels_folder, val_labels_folder, test_labels_folder]:
        os.makedirs(folder, exist_ok=True)

    for box_name in tqdm(os.listdir(args.boxes)):
        box_path = os.path.join(args.boxes, box_name)
        box_basename = os.path.splitext(os.path.splitext(box_name)[0])[0]

        with open(box_path, 'r') as f:
            label_data = json.load(f)

        lab_path = os.path.join(train_labels_folder, '{}.txt'.format(box_basename))
        if box_basename in val_images_basenames:
            lab_path = os.path.join(val_labels_folder, '{}.txt'.format(box_basename))
            assert os.path.exists(os.path.join(val_images_folder, '{}.png'.format(box_basename)))
        elif box_basename in test_images_basenames:
            lab_path = os.path.join(test_labels_folder, '{}.txt'.format(box_basename))
            assert os.path.exists(os.path.join(test_images_folder, '{}.png'.format(box_basename)))
        else:
            assert os.path.exists(os.path.join(train_images_folder, '{}.png'.format(box_basename)))

        with open(lab_path, 'w') as f:
            h = label_data['img_size']['height']
            w = label_data['img_size']['width']

            for detection in label_data['bb_objects']:
                category_id = get_class_id(detection['class'])
                x1 = detection['x1']
                y1 = detection['y1']
                x2 = detection['x2']
                y2 = detection['y2']

                cx = ((x2 + x1) / 2) / w
                cy = ((y2 + y1) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(
                    '{} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(
                        category_id,
                        cx, cy, bw, bh
                    )
                )
