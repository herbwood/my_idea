import os
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import dataset_utils

class CrowdHumanDataset(Dataset):

    def __init__(self, root_dir, phase, transform, class_names=['background', 'person']):
        
        self.phase = phase
        self.transform = transform
        self.class_names = class_names

        self.source = os.path.join(root_dir, f"annotation_{phase}.odgt")
        self.image_dir = os.path.join(root_dir, 'Images')

        self.records = dataset_utils.load_json_lines(self.source)
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self._load_record(self.records[index])

    def _load_record(self, record):

        image_path = os.path.join(self.image_dir, record["ID"] + ".jpg")
        image = dataset_utils.load_img(image_path)
        h = image.shape[0]
        w = image.shape[1]

        gtboxes = dataset_utils.load_gt(record, 'gtboxes', 'fbox', self.class_names)
        keep = (gtboxes[:, 2] >= 0) * (gtboxes[:, 3] >= 0)
        gtboxes = gtboxes[keep, :]
        gtboxes[:, 2:4] += gtboxes[:, :2]

        nb_gtboxes = gtboxes.shape[0]

        if self.transform is not None:
            image = self.transform(image)

        return image
        
    def merge_batch(self, data):
        
        # image
        images = [it[0] for it in data]
        gt_boxes = [it[1] for it in data]
        im_info = np.array([it[2] for it in data])

        # image height, width 
        batch_height = np.max(im_info[:, 3])
        batch_width = np.max(im_info[:, 4])

        # pad and resize images 
        padded_images = [self.pad_image(im, batch_height, batch_width, self.config.image_mean) for im in images]
        t_height, t_width, scale = self.target_size(batch_height, batch_width, self.short_size, self.max_size)
        # INTER_CUBIC, INTER_LINEAR, INTER_NEAREST, INTER_AREA, INTER_LANCZOS4
        resized_images = np.array([cv2.resize(
                im, (t_width, t_height), interpolation=cv2.INTER_LINEAR) for im in padded_images])

        resized_images = resized_images.transpose(0, 3, 1, 2)
        images = torch.tensor(resized_images).float()

        # ground_truth
        ground_truth = []
        for it in gt_boxes:
            gt_padded = np.zeros((self.config.max_boxes_of_image, self.config.nr_box_dim))
            it[:, 0:4] *= scale
            max_box = min(self.config.max_boxes_of_image, len(it))

            # scaled, padded된 gt box를 가져옴 
            gt_padded[:max_box] = it[:max_box]
            ground_truth.append(gt_padded)

        ground_truth = torch.tensor(ground_truth).float()

        # im_info
        im_info[:, 0] = t_height
        im_info[:, 1] = t_width
        im_info[:, 2] = scale
        im_info = torch.tensor(im_info)

        # gt box의 수가 2 미만인 경우... 
        if max(im_info[:, -1] < 2):
            return None, None, None
        else:
            return images, ground_truth, im_info

    def pad_image(img : np.ndarray, 
            height : int, 
            width : int, 
            mean_value : np.ndarray):

        o_h, o_w, _ = img.shape
        margins = np.zeros(2, np.int32)

        assert o_h <= height

        margins[0] = height - o_h
        img = cv2.copyMakeBorder(img, 0, margins[0], 0, 0, cv2.BORDER_CONSTANT, value=0)
        img[o_h:, :, :] = mean_value

        assert o_w <= width

        margins[1] = width - o_w
        img = cv2.copyMakeBorder(img, 0, 0, 0, margins[1], cv2.BORDER_CONSTANT, value=0)
        img[:, o_w:, :] = mean_value

        return img

    def target_size(height, width, short_size, max_size):
        # Rescale maintaing aspect ratio

        im_size_min = np.min([height, width])
        im_size_max = np.max([height, width])
        scale = (short_size + 0.0) / im_size_min

        if scale * im_size_max > max_size:
            scale = (max_size + 0.0) / im_size_max
        t_height, t_width = int(round(height * scale)), int(round(width * scale))

        return t_height, t_width, scale


if __name__ == "__main__":
    root_dir = "../dataset/crowd_human/"
    phase = 'train'
    transform = None

    crowdhuman = CrowdHumanDataset(root_dir, phase, transform)
    loader = DataLoader(crowdhuman, drop_last=True, batch_size=2, 
                shuffle=True, pin_memory=True, num_workers=2, collate_fn=crowdhuman.merge_batch)
    print(len(loader))

