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

import os
import argparse
import glob
import sys
from typing import List

import yaml
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--dst_model', required=True, help='averaged model')
    parser.add_argument('--src_path',
                        required=True,
                        help='src model path for average')
    parser.add_argument('--val_best',
                        action="store_true",
                        help='select best by validation loss')
    parser.add_argument('--test_best',
                        action="store_true",
                        help='select best by test acc from step_*.yaml')
    parser.add_argument('--num',
                        default=5,
                        type=int,
                        help='nums for averaged model')
    parser.add_argument('--min_epoch',
                        default=0,
                        type=int,
                        help='min epoch used for averaging model')
    parser.add_argument('--max_epoch',
                        default=sys.maxsize,
                        type=int,
                        help='max epoch used for averaging model')
    parser.add_argument('--min_step',
                        default=0,
                        type=int,
                        help='min step used for averaging model')
    parser.add_argument('--max_step',
                        default=sys.maxsize,
                        type=int,
                        help='max step used for averaging model')
    parser.add_argument('--mode',
                        default="hybrid",
                        choices=["hybrid", "epoch", "step"],
                        type=str,
                        help='average mode')

    parser.add_argument('--avg_ids',
                        default="",
                        type=str,
                        help='Comma-separated list of checkpoint ids/paths to average. '
                             'If a token is not an absolute path, it will be joined with --src_path. '
                             'If it does not end with .pt, .pt will be appended.')

    args = parser.parse_args()
    if args.val_best and args.test_best:
        raise ValueError('Use only one of --val_best or --test_best.')
    print(args)
    return args


def _resolve_avg_ids(avg_ids_str: str, src_path: str) -> List[str]:
    """Turn a comma-separated string into a concrete list of .pt file paths."""
    tokens = [t.strip() for t in avg_ids_str.split(',') if t.strip()]
    paths = []
    for t in tokens:
        candidate = t
        if not os.path.isabs(candidate):
            if not candidate.endswith('.pt'):
                candidate = candidate + '.pt'
            candidate = os.path.join(src_path, candidate)
        else:
            if not candidate.endswith('.pt'):
                candidate = candidate + '.pt'
        if not os.path.exists(candidate):
            raise FileNotFoundError(f'Checkpoint not found: {candidate}')
        paths.append(candidate)
    if len(paths) < 2:
        raise ValueError('--avg_ids requires at least 2 checkpoints.')
    return paths


def _collect_yaml_paths(src_path: str, mode: str) -> List[str]:
    if mode == "hybrid":
        yamls = glob.glob(os.path.join(src_path, '*.yaml'))
        yamls = [
            f for f in yamls
            if not (os.path.basename(f).startswith('train')
                    or os.path.basename(f).startswith('init'))
        ]
    elif mode == "step":
        yamls = glob.glob(os.path.join(src_path, 'step_*.yaml'))
    else:
        yamls = glob.glob(os.path.join(src_path, 'epoch_*.yaml'))
    return yamls


def _filter_by_range(record_epoch: int, record_step: int,
                     min_epoch: int, max_epoch: int,
                     min_step: int, max_step: int) -> bool:
    return (min_epoch <= record_epoch <= max_epoch) and (min_step <= record_step <= max_step)


def _select_by_val_best(yamls: List[str], num: int,
                        min_epoch: int, max_epoch: int,
                        min_step: int, max_step: int) -> List[str]:
    candidates = []
    for y in yamls:
        try:
            with open(y, 'r') as f:
                d = yaml.load(f, Loader=yaml.FullLoader)
            loss = d['loss_dict']['loss']
            epoch = d['epoch']
            step = d['step']
            tag = d['tag']
            if _filter_by_range(epoch, step, min_epoch, max_epoch, min_step, max_step):
                candidates.append((epoch, step, loss, tag))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[2])  # lower loss is better
    print("best val (epoch, step, loss, tag) =", str(candidates[:num]))
    return [os.path.join(os.path.abspath(os.path.dirname(yamls[0])), f'{c[-1]}.pt') for c in candidates[:num]]


def _select_by_test_best(yamls: List[str], num: int,
                         min_epoch: int, max_epoch: int,
                         min_step: int, max_step: int) -> List[str]:
    candidates = []
    for y in yamls:
        try:
            with open(y, 'r') as f:
                d = yaml.load(f, Loader=yaml.FullLoader)
            # Must be step_* yaml to have test metrics reliably
            tag = d.get('tag', '')
            if not tag.startswith('step_'):
                continue
            epoch = d['epoch']
            step = d['step']
            test_acc = d['test_loss_dict']['acc']
            if _filter_by_range(epoch, step, min_epoch, max_epoch, min_step, max_step):
                candidates.append((epoch, step, test_acc, tag))
        except Exception:
            continue
    candidates.sort(key=lambda x: x[2], reverse=True)
    print("best test (epoch, step, acc, tag) =", str(candidates[:num]))
    base = os.path.abspath(os.path.dirname(yamls[0])) if yamls else ''
    return [os.path.join(base, f'{c[-1]}.pt') for c in candidates[:num]]


def _average_checkpoints(paths: List[str], dst: str):
    print(paths)
    avg = {}
    num = len(paths)
    assert num >= 2, "Need at least two checkpoints to average"
    for path in paths:
        print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        for k in states.keys():
            if k not in avg:
                avg[k] = states[k].clone()
            else:
                avg[k] += states[k]
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.true_divide(avg[k], num)
    print('Saving to {}'.format(dst))
    torch.save(avg, dst)


def main():
    args = get_args()

    if args.avg_ids:
        path_list = _resolve_avg_ids(args.avg_ids, args.src_path)
        print(f'Using explicitly specified checkpoints ({len(path_list)}):')
        for p in path_list:
            print('  -', p)
        _average_checkpoints(path_list, args.dst_model)
        return

    if args.val_best or args.test_best:
        yamls = _collect_yaml_paths(args.src_path, args.mode)
        if args.test_best:
            path_list = _select_by_test_best(
                yamls, args.num, args.min_epoch, args.max_epoch, args.min_step, args.max_step
            )
        else:
            path_list = _select_by_val_best(
                yamls, args.num, args.min_epoch, args.max_epoch, args.min_step, args.max_step
            )
    else:
        path_list = glob.glob(os.path.join(args.src_path, '[!init]*.pt'))
        path_list = sorted(path_list, key=os.path.getmtime)
        path_list = path_list[-args.num:]

    _average_checkpoints(path_list, args.dst_model)


if __name__ == '__main__':
    main()
