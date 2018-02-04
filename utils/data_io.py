#!/usr/bin/python
# -*- coding: utf-8 -*-
# 
# Originally developed by Chris Choy <chrischoy@ai.stanford.edu>
# Updated by Haozhe Xie <cshzxie@gmail.com>
# 
# CHANGELOG:
# - 2018/02/04: Removed unused functions

import os
import json

from collections import OrderedDict
from datetime import datetime as dt

def category_model_id_pair(dataset_taxonomy_file, dataset_query_path, dataset_portion=[]):
    '''Load category, model names from a ShapeNet dataset.'''
    def model_names(model_path):
        """ Return model names"""
        model_names = [name for name in os.listdir(model_path)
                       if os.path.isdir(os.path.join(model_path, name))]
        return sorted(model_names)

    category_name_pair = []  # full path of the objs files

    cats = json.load(open(dataset_taxonomy_file))
    cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))

    for k, cat in cats.items():  # load by categories
        model_path = os.path.join(dataset_query_path, cat['id'])
        models     = model_names(model_path)
        num_models = len(models)
        portioned_models = models[int(num_models * dataset_portion[0]):int(num_models *
                                                                           dataset_portion[1])]
        category_name_pair.extend([(cat['id'], model_id) for model_id in portioned_models])

    print('[INFO] %s Load data categories from %s' % (dt.now(), dataset_taxonomy_file))
    return category_name_pair

def get_voxel_file(voxel_path, category, model_id):
    return voxel_path % (category, model_id)

def get_rendering_file(rendering_path, category, model_id, rendering_id):
    return os.path.join(rendering_path % (category, model_id), '%02d.png' % rendering_id)