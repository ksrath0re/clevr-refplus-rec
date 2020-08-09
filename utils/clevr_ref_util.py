import os, json
import numpy as np
import cv2

class clevr_ref_util:
    def __init__(self, scene_file, refexp_path, num_refexp=-1):
        self.scene_file = scene_file
        self.refexp_path = refexp_path
        self.num_refexp = num_refexp
        self.scenes = None
        self.exps = None

    def load_scene_refexp(self):
        print('loading scene.json...')
        scenes = json.load(open(self.scene_file))
        self.scenes = scenes['scenes']
        print('loading refexp.json...')
        if self.num_refexp != -1:
            self.exps = json.load(open(self.refexp_path))['refexps'][:self.num_refexp]
        else:
            self.exps = json.load(open(self.refexp_path))['refexps'][:]
        print('loading json done')

        self.imgid_scenes={}
        for sce in self.scenes:
          img_id = sce['image_index']
          self.imgid_scenes[img_id] = sce

    def get_mask_from_refexp(self, refexp, height=-1, width=-1):
        sce = self.get_scene_of_refexp(refexp)
        obj_list = self.get_refexp_output_objectlist(refexp)

        heatmap = np.zeros((320,480))

        def from_imgdensestr_to_imgarray(imgstr):
            img = []
            cur = 0
            for num in imgstr.split(','):
                num = int(num)
                img += [cur]*num;
                cur = 1-cur
            img = np.asarray(img).reshape((320,480))
            return img

        for objid in obj_list:
            obj_mask = sce['obj_mask'][str(objid+1)]
            mask_img = from_imgdensestr_to_imgarray(obj_mask)
            heatmap += mask_img
        if height !=-1 and width !=-1:
            heatmap = cv2.resize(heatmap, (width, height))
        return heatmap


    def get_scene_of_refexp(self, exp):
        image_index = exp['image_index']
        sce = self.imgid_scenes[image_index]
        return sce


    def get_refexp_output_objectlist(self, exp):
        prog = exp['program']
        image_filename = exp['image_filename']
        last = prog[-1]
        obj_list = last['_output']
        return obj_list
