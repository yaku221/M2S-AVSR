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
import datetime
import logging
import sys
from contextlib import nullcontext

# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from wenet.utils.common import StepTimer

from wenet.utils.train_utils_avsr import (wenet_join, batch_forward, batch_backward,
                                     update_parameter_and_lr, log_per_step,
                                     save_model)


class Executor:

    def __init__(self,
                 global_step: int = 0,
                 device: torch.device = torch.device("cpu")):
        self.step = global_step + 1
        self.train_step_timer = None
        self.cv_step_timer = None
        self.device = device

    def train(self, model, optimizer, scheduler, train_data_loader,
              cv_data_loader, writer, configs, scaler, group_join,
              test_data_loader=None):
        ''' Train one epoch
        '''
        if self.train_step_timer is None:
            self.train_step_timer = StepTimer(self.step)
        model.train()
        
        # use batch stats from ckpt
        self._set_batch_norm_eval(model, configs)          
            
        info_dict = copy.deepcopy(configs)
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_context = model.join
        else:
            model_context = nullcontext

        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                if wenet_join(group_join, info_dict):
                    break

                if batch_dict["target_lengths"].size(0) == 0:
                    continue

                context = None
                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict.get("train_engine", "torch_ddp") in [
                        "torch_ddp", "torch_fsdp"
                ] and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler,
                                              info_dict, self.device)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer,
                                                    scheduler, scaler,
                                                    info_dict)
                # import ipdb; ipdb.set_trace()
                # write training: tensorboard && log
                log_per_step(writer, info_dict, timer=self.train_step_timer)
                save_interval = info_dict.get('save_interval', sys.maxsize)
                if (self.step +
                        1) % save_interval == 0 and self.step != 0 and (
                            batch_idx + 1) % info_dict["accum_grad"] == 0:
                    import torch.distributed as dist
                    # Ensure all ranks start CV at the same time in step mode
                    dist.barrier()
                    # import ipdb; ipdb.set_trace()
                    
                    cv_loss_dict = self.cv(model, cv_data_loader, configs, writer, tag='CV')
                    model.train()                    
                    # use batch stats from ckpt
                    self._set_batch_norm_eval(model, configs)
                    
                    test_loss_dict = None
                    if configs.get('val_test_data', False) and (test_data_loader is not None):
                        test_loss_dict = self.cv(model, test_data_loader, configs, writer, tag='TEST')
                        model.train()
                        self._set_batch_norm_eval(model, configs)  
                         
                    # import ipdb; ipdb.set_trace()
                    info_cv = {
                        "tag": f"step_{self.step}",
                        "step": self.step,
                        "batch_idx": 0,  
                        "epoch": info_dict.get("epoch", 0),
                        "lrs": [group['lr'] for group in optimizer.param_groups],
                        "loss_dict": cv_loss_dict,
                    }
                    log_per_step(writer, info_cv)  # default eval_prefix='cv'

                    if test_loss_dict is not None:
                        info_test = dict(info_cv)
                        info_test["loss_dict"] = test_loss_dict
                        info_test["eval_prefix"] = "test"
                        log_per_step(writer, info_test)

                    info_dict.update({
                        "tag": f"step_{self.step}",
                        "loss_dict": cv_loss_dict,        
                        "test_loss_dict": test_loss_dict, 
                        "save_time": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "lrs": [group['lr'] for group in optimizer.param_groups]
                    })
                    
                    save_model(model, info_dict)
                    # log_per_step(writer, info_dict)
                    log_per_step(writer=None, info_dict=info_dict)
                    # Ensure all ranks start Train at the same time in step mode
                    dist.barrier()
                self.step += 1 if (batch_idx +
                                   1) % info_dict["accum_grad"] == 0 else 0

    def cv(self, model, cv_data_loader, configs, writer, tag='CV'):
        ''' Cross validation on
        '''
        if self.cv_step_timer is None:
            self.cv_step_timer = StepTimer(0.0)
        else:
            self.cv_step_timer.last_iteration = 0.0
            
        tag = str(tag).upper()
        tb_prefix = 'cv' if tag == 'CV' else 'test'
        
        model.eval()
        info_dict = copy.deepcopy(configs)
        num_seen_utts, loss_dict, total_acc = 1, {}, []  # avoid division by 0

        # import ipdb; ipdb.set_trace()
        # -------- for log attn_gate.tanh(),ff_gate.tanh() --------
        gate_stats = {}
        if getattr(model, "decoder", None) is not None and \
        any(getattr(l, "add_gated_x_attn", False) for l in model.decoder.decoders):
            wanted = configs.get("gate_log_layers", None)
            if wanted is None:
                total = len(model.decoder.decoders) 
                wanted = self.choose_gate_log_layers(total, 5)
            gate_stats = self._collect_visual_gate_stats(model, layers=wanted)

        # -------- for track_norm --------
        sum_x = sum_xv_pre = sum_xv_post = 0.0
        n_x = 0

        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(cv_data_loader):
                info_dict["tag"] = tag
                info_dict["step"] = self.step
                info_dict["batch_idx"] = batch_idx
                info_dict[f"{tb_prefix}_step"] = batch_idx
                info_dict["track_norm"] = True

                num_utts = batch_dict["target_lengths"].size(0)
                if num_utts == 0:
                    continue
                # import ipdb; ipdb.set_trace()
                info_dict = batch_forward(model, batch_dict, None, info_dict,
                                          self.device)
                _dict = info_dict["loss_dict"]

                num_seen_utts += num_utts
                total_acc.append(_dict['th_accuracy'].item(
                ) if _dict.get('th_accuracy', None) is not None else 0.0)
                
                # import ipdb; ipdb.set_trace()
                # ============ for track_norm ============
                if "x_norm" in _dict:
                    sum_x += float(_dict["x_norm"])
                    sum_xv_pre += float(_dict.get("x_v_norm_pre", 0.0))
                    sum_xv_post += float(_dict.get("x_v_norm_post", 0.0))
                    n_x += 1
                    
                for loss_name, loss_value in _dict.items():
                    if loss_value is not None and "loss" in loss_name \
                            and torch.isfinite(loss_value):
                        loss_value = loss_value.item()
                        loss_dict[loss_name] = loss_dict.get(loss_name, 0) + \
                            loss_value * num_utts
                # write cv: log
                log_per_step(writer=None,
                             info_dict=info_dict,
                             timer=self.cv_step_timer)
        for loss_name, loss_value in loss_dict.items():
            loss_dict[loss_name] = loss_dict[loss_name] / num_seen_utts
        loss_dict["acc"] = sum(total_acc) / len(total_acc)
        
        # average track_norm
        if n_x > 0:
            loss_dict["x_norm"] = sum_x / n_x
            loss_dict["x_v_norm_pre"] = sum_xv_pre / n_x
            loss_dict["x_v_norm_post"] = sum_xv_post / n_x

        # add attn_gate.tanh(),ff_gate.tanh()
        loss_dict.update(gate_stats)
        
        return loss_dict
    
    @staticmethod
    def _collect_visual_gate_stats(model, layers=None):
        """return: attn_gate.tanh(),ff_gate.tanh()"""
        stats = {}
        dec = getattr(model, "decoder", None)
        if dec is None:
            return stats
        
        layers_list = getattr(dec, "decoders", [])
        n = len(layers_list)
        
        if layers is None:
            idx_iter = range(n)
        else:
            idx_iter = [i for i in layers if 0 <= i < n]

        for i in idx_iter:
            layer = layers_list[i]
            if getattr(layer, "add_gated_x_attn", False) and hasattr(layer, "attn_gate"):
                g_attn = torch.tanh(layer.attn_gate.detach()).item()
                g_ff   = torch.tanh(layer.ff_gate.detach()).item()
                stats[f"attn_gate_layer_{i}"] = g_attn
                stats[f"ff_gate_layer_{i}"] = g_ff
        return stats
    
    @staticmethod
    def choose_gate_log_layers(num_blocks: int, k: int = 5):
        if num_blocks <= 0:
            return []
        k = min(k, num_blocks)
        if k == 1:
            return [num_blocks - 1]
        idx = { round(i * (num_blocks - 1) / (k - 1)) for i in range(k) }
        idx.add(num_blocks - 1)
        return sorted(idx)
    
    @staticmethod
    def _set_batch_norm_eval(model, configs):
        freeze_video_batch_norm_stats = bool(configs.get('freeze_video_batch_norm_stats', False))
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'video_model'):
            if freeze_video_batch_norm_stats:
                model.encoder.video_model.eval()