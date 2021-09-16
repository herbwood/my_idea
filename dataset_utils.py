import os
import cv2
import json
import numpy as np

def load_json_lines(fpath : str):

    assert os.path.exists(fpath)

    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines]

    return records

def load_img(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    return img

def load_gt(dict_input, 
            key_name, 
            key_box, 
            class_names):

    assert key_name in dict_input

    if len(dict_input[key_name]) < 1:
        return np.empty([0, 5])
    else:
        assert key_box in dict_input[key_name][0]

    bbox = []

    for rb in dict_input[key_name]:
        if rb['tag'] in class_names:
            tag = class_names.index(rb['tag']) # background = 0, person = 1
        else:
            tag = -1
        if 'extra' in rb:
            if 'ignore' in rb['extra']:
                if rb['extra']['ignore'] != 0:
                    tag = -1
        bbox.append(np.hstack((rb[key_box], tag)))

    bboxes = np.vstack(bbox).astype(np.float64)


    return bboxes