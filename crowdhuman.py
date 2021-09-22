import os
import cv2
from collections import defaultdict

import numpy as np
import mmcv_utils
from torch.utils.data import Dataset

class CrowdHumanDataset(Dataset):

    CLASSES = None

    def __init__(self, data_root, img_prefix, ann_file, transform=None, classes=None, test_mode=False):
        self.data_root = data_root
        self.img_prefix = img_prefix 
        self.ann_file = ann_file
        self.transform = transform 
        self.CLASSES = self.get_classes(classes)
        self.test_mode = test_mode

        if self.data_root is not None:
            if not os.path.isabs(self.ann_file):
                self.ann_file = os.path.join(self.data_root, self.ann_file)
            if not os.path.isabs(self.img_prefix):
                self.img_path = os.path.join(self.data_root, self.img_prefix)

        # load annotations 
        self.data_infos = self.load_annotations(self.ann_file)

        # filter images too small and containing no annotations 
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        data = mmcv_utils.load(ann_file)
        results = defaultdict(lambda: {'ann': defaultdict(list)})
        image_data = {v['id']: v for v in data['images']}

        for annotation in data['annotations']:
            image_id = annotation['image_id']
            results[image_id]['filename'] = image_data[image_id]['file_name']
            results[image_id]['width'] = image_data[image_id]['width']
            results[image_id]['height'] = image_data[image_id]['height']
            bbox = annotation['bbox']
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            results[image_id]['ann']['bboxes_ignore' if annotation['iscrowd'] else 'bboxes'].append(bbox)
            results[image_id]['ann']['labels_ignore' if annotation['iscrowd'] else 'labels'].append(0)
        results = list(results.values())

        for annotation in results:
            annotation['ann']['bboxes'] = np.array(annotation['ann']['bboxes'], dtype=np.float32)
            if not len(annotation['ann']['bboxes']):
                annotation['ann']['bboxes'] = np.zeros((0, 4), dtype=np.float32)
            annotation['ann']['labels'] = np.array(annotation['ann']['labels'], dtype=np.int64)
            annotation['ann']['bboxes_ignore'] = np.array(annotation['ann']['bboxes_ignore'], dtype=np.float32)
            if not len(annotation['ann']['bboxes_ignore']):
                annotation['ann']['bboxes_ignore'] = np.zeros((0, 4), dtype=np.float32)
            annotation['ann']['labels_ignore'] = np.array(annotation['ann']['labels_ignore'], dtype=np.int64)
        return results


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def _filter_imgs(self, min_size=32):
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.transform is not None:
            return self.transform(results)
        return results 

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.transform is not None:
            return self.transform(results)
        return results

    @classmethod 
    def get_classes(cls, classes=None):
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            class_names = mmcv_utils.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        
        return class_names 

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 iou_thr=0.5):
        # annotations to brambox
        true_df = defaultdict(list)
        for img_info in self.data_infos:
            bboxes = np.concatenate((img_info['ann']['bboxes'], img_info['ann']['bboxes_ignore']), axis=0)
            labels = np.concatenate((img_info['ann']['labels'], img_info['ann']['labels_ignore']), axis=0)
            ignores = [False] * len(img_info['ann']['bboxes']) + [True] * len(img_info['ann']['bboxes_ignore'])
            for bbox, label, ignore in zip(bboxes, labels, ignores):
                true_df['image'].append(img_info['filename'])
                true_df['class_label'].append(label)
                true_df['id'].append(0)
                true_df['x_top_left'].append(bbox[0])
                true_df['y_top_left'].append(bbox[1])
                true_df['width'].append(bbox[2] - bbox[0])
                true_df['height'].append(bbox[3] - bbox[1])
                true_df['ignore'].append(ignore)
        true_df = pd.DataFrame(true_df)
        true_df['image'] = true_df['image'].astype('category')

        # results to brambox
        predicted_df = defaultdict(list)
        for i, image_results in enumerate(results):
            for j, class_detection in enumerate(image_results):
                for detection in class_detection:
                    predicted_df['image'].append(self.data_infos[i]['filename'])
                    predicted_df['class_label'].append(j)
                    predicted_df['id'].append(0)
                    predicted_df['x_top_left'].append(detection[0])
                    predicted_df['y_top_left'].append(detection[1])
                    predicted_df['width'].append(detection[2] - detection[0])
                    predicted_df['height'].append(detection[3] - detection[1])
                    predicted_df['confidence'].append(detection[4])
        predicted_df = pd.DataFrame(predicted_df)
        predicted_df['image'] = predicted_df['image'].astype('category')

        pr = brambox.stat.pr(predicted_df, true_df, iou_thr)
        ap = brambox.stat.ap(pr)
        mr_fppi = brambox.stat.mr_fppi(predicted_df, true_df, iou_thr)
        lamr = brambox.stat.lamr(mr_fppi)
        eval_results = {
            'gts': len(true_df[~true_df['ignore']]),
            'dets': len(predicted_df),
            'recall': pr['recall'].values[-1],
            'mAP': ap,
            'mMR': lamr
        }
        print(str(eval_results), logger)
        return eval_results

if __name__ == "__main__":
    crowdhuman = CrowdHumanDataset(data_root='../dataset/crowd_human/', 
                                   img_prefix='Images',
                                   ann_file='annotation_full_val.json', 
                                   transform=None, 
                                   classes=None, 
                                   test_mode=False)
    print(crowdhuman[0])
    print(crowdhuman[0]['img_info'].keys())
    print(crowdhuman[0]['ann_info'].keys())

    """
    {'img_info': 
        {'ann': defaultdict(<class 'list'>, 
            {'bboxes': array([[ 72., 202., 235., 705.],
                                [199., 180., 343., 679.],
                                [310., 200., 472., 697.],
                                [417., 182., 556., 700.],
                                [429., 171., 653., 682.],
                                [543., 178., 805., 748.]], dtype=float32), 
             'labels': array([0, 0, 0, 0, 0, 0], dtype=int64), 
             'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 
             'labels_ignore': array([], dtype=int64)}), 
         'filename': '273271,c9db000d5146c15.jpg', 
         'width': 800, 
         'height': 600
        }, 
             
             
    'ann_info': defaultdict(<class 'list'>, 
            {'bboxes': array([[ 72., 202., 235., 705.],
                                [199., 180., 343., 679.],
                                [310., 200., 472., 697.],
                                [417., 182., 556., 700.],
                                [429., 171., 653., 682.],
                                [543., 178., 805., 748.]], dtype=float32), 
            'labels': array([0, 0, 0, 0, 0, 0], dtype=int64), 
            'bboxes_ignore': array([], shape=(0, 4), dtype=float32), 
            'labels_ignore': array([], dtype=int64)
            }
        )
    }
    """