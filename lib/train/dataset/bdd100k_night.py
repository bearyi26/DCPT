import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
from collections import OrderedDict
from lib.train.admin import env_settings
from collections import defaultdict
import time,json

class BDD100K_Night(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        root = env_settings().bdd100k_dir if root is None else root
        super().__init__('bdd100k_night', root, image_loader)

        self.img_pth = os.path.join(root, 'images/')
        self.anno_path = os.path.join(root, 'annotations/bdd100k_night.json')

        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not self.anno_path == None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(self.anno_path, 'r') as f:
                dataset = json.load(f)
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
        self.sequence_list = self._get_sequence_list()
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))


    #得到序列
    def _get_sequence_list(self):
        anns = {}
        for picture in self.dataset:
            for box in picture['labels']:
                anns[box['id']] = box
                anns[box['id']]['name'] = picture['name']
        self.anns = anns

        #anns对应的是每一个框
        seq_list = list(anns.keys())

        return seq_list

    def _get_anno(self, seq_id):
        anno = self.anns[self.sequence_list[seq_id]]
        return anno


    #得到图片帧
    def _get_frames(self, seq_id):
        path = self.anns[self.sequence_list[seq_id]]['name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    #得到每一帧的bounding box
    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)

        x = anno['box2d']['x1']
        y = anno['box2d']['y1']
        width = anno['box2d']['x2'] - anno['box2d']['x1']
        height = anno['box2d']['y2'] - anno['box2d']['y1']

        bbox = torch.Tensor([x,y,width,height]).view(1, 4)

        '''v0.4 BDD100K_Night avoid too small bounding boxes'''
        valid = (bbox[:, 2] > 50) & (bbox[:, 3] > 50)

        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def is_video_sequence(self):
        return False

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # BDD100K is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

    def get_name(self):
        return 'bdd100k_night'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.anns[self.sequence_list[seq_id]]['category']
            object_meta = OrderedDict({'object_class_name': cat_dict_current,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta
