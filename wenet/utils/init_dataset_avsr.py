# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
# Copyright (c) 2025 Fei Su (shinji721@outlook.com)
#
# Modified from the original WeNet implementation by Fei Su, 2025.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Optional
from wenet.dataset.dataset_avsr import Dataset_AV
from wenet.text.base_tokenizer import BaseTokenizer


def init_avsr_dataset(data_type,
                    audio_data_list_file,
                    video_data_list_file,
                    tokenizer: Optional[BaseTokenizer] = None,
                    conf=None,
                    partition=True):
    return Dataset_AV(data_type, audio_data_list_file, video_data_list_file, 
                      tokenizer, conf, partition)    
    
    
def init_dataset_avsr(dataset_type,
                 data_type,
                 audio_data_list_file,
                 video_data_list_file,
                 tokenizer: Optional[BaseTokenizer] = None,
                 conf=None,
                 partition=True,
                 split='train'):
    assert dataset_type in ['avsr']
    
    conf['train'] = True
    conf['split'] = 'train'
    if split != 'train':
        cv_conf = copy.deepcopy(conf)
        cv_conf['cycle'] = 1
        cv_conf['speed_perturb'] = False
        cv_conf['spec_aug'] = False
        cv_conf['spec_sub'] = False
        cv_conf['spec_trim'] = False
        cv_conf['shuffle'] = False
        cv_conf['list_shuffle'] = False
        cv_conf['list_shuffle'] = False
        cv_conf['train'] = False
        cv_conf['split'] = 'cv'
        if split == 'test':
            cv_conf['split'] = 'test'
        conf = cv_conf

    # import ipdb;ipdb.set_trace()
    if dataset_type == 'avsr':
        return init_avsr_dataset(data_type, audio_data_list_file, video_data_list_file, 
                                 tokenizer, conf, partition)
    else:
        raise ValueError(f"Unsupported dataset_type={dataset_type}")



