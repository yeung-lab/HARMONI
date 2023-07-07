from collections import defaultdict, Counter

import numpy as np

class Results(object):
    def __init__(self):
        self.results = {}
        self.results_smoothed = {}
        self.adult_bottom = defaultdict(list)
        self.infant_bottom = defaultdict(list)
        self.cam_params = {}
    
    def update_results(self, idxs, results):
        for idx, res in zip(idxs, results):
            self.results[idx] = res

    def update_scene(self, img_name, ankle_locations, cam_params, body_type):
        if not hasattr(self, 'adult_bottom'):
            self.adult_bottom = defaultdict(list)
            self.infant_bottom = defaultdict(list)
            self.cam_params = {}
        if body_type == 'adult':
            self.adult_bottom[img_name].extend(ankle_locations)
        elif body_type == 'infant':
            self.infant_bottom[img_name].extend(ankle_locations)
        else:
            raise Exception("body type {} not recognized".format(body_type))
        self.cam_params[img_name] = cam_params
