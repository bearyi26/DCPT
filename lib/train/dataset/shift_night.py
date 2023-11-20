import os
import os.path
import torch
import numpy as np
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import json

class SHIFT_Night(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        """
        SHIFT_NIGHT Dataset
        """
        root = env_settings().shift_dir if root is None else root
        super().__init__('shift_night', root, image_loader)

        sequence_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        sequence_path = os.path.join(sequence_path, 'data_specs', 'shift_info_1fps.json')
        with open(sequence_path, 'r') as f:
            info = json.load(f)
        self.info = info

        self.sequence_list = self._build_sequence_list()

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def _build_sequence_list(self):
        sequence_list = [sequence for sequence in self.info.keys()]
        return sequence_list

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        video_name = seq_name.split('/')[0]
        return os.path.join(self.root, video_name), seq_name

    def _get_frame_path(self, seq_path, seq_name, frame_id):
        frame = self.info[seq_name]['frame'][frame_id]
        return os.path.join(seq_path, frame)    # frames extracted from info.json

    def _get_frame(self, seq_path, seq_name, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, seq_name, frame_id))

    def _read_bb_anno(self, seq_path, seq_name):
        bbox_all = []
        for bbox in self.info[seq_name]['box2d']:
            x = bbox['x1']
            y = bbox['y1']
            width = bbox['x2'] - bbox['x1']
            height = bbox['y2'] - bbox['y1']
            bbox_np = np.array([[x,y,width,height]])
            bbox_all.append(bbox_np)
            bbox_all_np = np.concatenate([bbox for bbox in bbox_all],axis=0)
        return torch.tensor(bbox_all_np)

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        seq_path, seq_name = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path, seq_name)

        '''v0.4 Shift avoid too small bounding boxes'''
        valid = (bbox[:, 2] > 50) & (bbox[:, 3] > 50)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_name(self):
        return 'shift_night'

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path, seq_name = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, seq_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': self.info[seq_name]['category'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta