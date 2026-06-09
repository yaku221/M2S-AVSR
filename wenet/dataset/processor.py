# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
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


import io
import os
import json
import math
from subprocess import PIPE, Popen
from urllib.parse import urlparse
from langid.langid import LanguageIdentifier, model
import logging
import librosa
import random
import numpy as np
from scipy.io import wavfile

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from wenet.text.base_tokenizer import BaseTokenizer

torchaudio.utils.sox_utils.set_buffer_size(16500)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)

logging.getLogger('langid').setLevel(logging.INFO)

NOISE_CACHE = {}  # {noise_list_path: [wav_path, ...]}


try:
    cpu_info = os.popen("lscpu | grep 'Vendor ID'").read()
    # 0x48 --> HiSilicon
    if (cpu_info.rstrip().split(" ")[-1] == "0x48"):
        # NOTE (MengqingCao): set number of threads in the subprocesses to 1
        # Why? There may be some operators ultilizing multi-threads in processor,
        # causing possibly deadlock in Kunpeng.
        # Similar issue in PyTorch: https://github.com/pytorch/pytorch/issues/45198
        torch.set_num_threads(1)
except Exception as ex:
    logging.warning('Failed to set number of thread in Kunpeng, \
        this may cause segmentfault while dataloading, \
        ignore this warning if you are not using Kunpeng')


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def parse_speaker(sample, speaker_dict):
    assert 'speaker' in sample
    speaker = sample['speaker']
    sample['speaker'] = speaker_dict.get(speaker, 0)
    return sample


def detect_language(sample, limited_langs):
    assert 'txt' in sample
    # NOTE(xcsong): Because language classification may not be very accurate
    #   (for example, Chinese being classified as Japanese), our workaround,
    #   given we know for certain that the training data only consists of
    #   Chinese and English, is to limit the classification results to reduce
    #   the impact of misclassification.
    lid.set_languages(limited_langs)
    # i.e., ('zh', 0.9999999909903544)
    sample['lang'] = lid.classify(sample['txt'])[0]
    return sample


def detect_task(sample):
    # TODO(xcsong): Currently, the task is hard-coded to 'transcribe'.
    #   In the future, we could dynamically determine the task based on
    #   the contents of sample. For instance, if a sample contains both
    #   'txt_en' and 'txt_zh', the task should be set to 'translate'.
    sample['task'] = "transcribe"
    return sample


def decode_wav(sample):
    """ Parse key/wav/txt from json line

        Args:
            sample: str, str is a json line has key/wav

        Returns:
            {key, wav, sample_rate, ...}
    """
    assert 'key' in sample
    assert 'wav' in sample
    wav_file = sample['wav']
    if isinstance(wav_file, str):
        with open(wav_file, 'rb') as f:
            wav_file = f.read()
    if 'start' in sample:
        assert 'end' in sample
        with io.BytesIO(wav_file) as file_obj:
            sample_rate = torchaudio.info(file_obj).sample_rate
            start_frame = int(sample['start'] * sample_rate)
            end_frame = int(sample['end'] * sample_rate)
            file_obj.seek(0)
            waveform, _ = torchaudio.load(file_obj,
                                          num_frames=end_frame - start_frame,
                                          frame_offset=start_frame)
    else:
        with io.BytesIO(wav_file) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
    # del wav_file
    del sample['wav']
    sample['wav'] = waveform  # overwrite wav
    sample['sample_rate'] = sample_rate
    return sample


def singal_channel(sample, channel=0):
    """ Choose a channel of sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            channel: target channel index

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'wav' in sample
    waveform = sample['wav']
    channel_nums = waveform.size(0)
    assert channel < channel_nums
    if channel_nums != 1:
        waveform = waveform[channel, :].unsqueeze(0)
    sample['wav'] = waveform
    return sample


def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def speed_perturb(sample, speeds=None):
    """ Apply speed perturb to the sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            speeds(List[float]): optional speed

        Returns:
            key, wav, label, sample_rate}
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    speed = random.choice(speeds)
    if speed != 1.0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        sample['wav'] = wav

    return sample


def add_noise(sample, noise_prob=0.0, noise_snr=0, noise_list_path=None):
    if (noise_prob <= 0.0) or (not noise_list_path):
        return sample
    if np.random.rand() > float(noise_prob):
        return sample

    assert 'wav' in sample and 'sample_rate' in sample
    clean = sample['wav']   # (1, N), torch.float32, normalized to [-1,1]
    sr = int(sample['sample_rate'])

    # import ipdb; ipdb.set_trace()
    if not (isinstance(clean, torch.Tensor) and clean.dim() == 2 and clean.size(0) == 1):
        return sample
    clean_np = clean.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (N,)

    try:
        if noise_list_path in NOISE_CACHE:
            noise_paths = NOISE_CACHE[noise_list_path]
        else:
            noise_paths = []
            with open(noise_list_path, 'r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue                    
                    p = line.split('\t')[0].strip()
                    if p:
                        noise_paths.append(p)
            NOISE_CACHE[noise_list_path] = noise_paths
    except Exception as e:
        msg = f"Add Noise Error: failed to read noise list {noise_list_path}: {repr(e)}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    if not noise_paths:
        msg = f"Add Noise Error: empty noise list after reading {noise_list_path}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    chosen_wav = np.random.choice(noise_paths)
    if not os.path.isabs(chosen_wav):
        chosen_wav = os.path.abspath(chosen_wav)
    if not os.path.exists(chosen_wav):
        msg = f"Load Noise File Error: noise file not found: {chosen_wav}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    try:
        n_sr, n_wav = wavfile.read(chosen_wav)  # int16/float
        if n_sr != sr:
            raise ValueError(f"The sample_rate of noise wav is {n_sr}")    
    except Exception as e:
        msg = f"Load Noise File Error: failed to read noise wav {chosen_wav}: {repr(e)}"
        logging.error(msg)
        raise 

    if n_wav.dtype == np.int16:
        noise_np = (n_wav.astype(np.float32) / 32768.0)
    else:
        noise_np = n_wav.astype(np.float32)
        
    if noise_np.ndim == 2 and noise_np.shape[1] > 1:
        noise_np = noise_np.mean(axis=1)

    noise_np = noise_np.flatten()

    len_clean_wav  = clean_np.shape[0]
    len_noise_wav = noise_np.shape[0]
    if len_noise_wav < len_clean_wav:
        ratio = int(np.ceil(len_clean_wav/len_noise_wav))
        noise_np = np.concatenate([noise_np for _ in range(ratio)])
        len_noise_wav = noise_np.shape[0]
    if len_noise_wav > len_clean_wav:
        start = 0  
        noise_np = noise_np[start: start + len_clean_wav]

    if isinstance(noise_snr, (tuple, list)):
        snr_db = float(np.random.randint(int(noise_snr[0]), int(noise_snr[1]) + 1))
    else:
        snr_db = float(noise_snr)

    eps = 1e-8
    clean_rms = np.sqrt(np.mean(clean_np**2) + eps)
    noise_rms = np.sqrt(np.mean(noise_np**2) + eps)
    if noise_rms < eps:
        msg = f"Add Noise Error: noise RMS ~ 0 for file {chosen_wav}"
        logging.error(msg)
        raise ValueError(msg)
    
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    noise_np = noise_np * (target_noise_rms / (noise_rms + eps))
    mixed = clean_np + noise_np

    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    sample['wav'] = torch.from_numpy(mixed).to(clean.device, dtype=clean.dtype).unsqueeze(0)
    return sample


def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  window_type="povey"):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate,
                      window_type=window_type)
    sample['feat'] = mat
    return sample


def compute_w2vbert_fbank(sample,
                          num_mel_bins=23,
                          frame_length=25,
                          frame_shift=10,
                          dither=0.0):
    """ Extract Pretrain w2vbert(4.5M hours) fbank
    """
    sample = compute_fbank(sample, num_mel_bins, frame_length, frame_shift,
                           dither)
    mat = sample['feat']
    std, mean = torch.std_mean(mat, dim=0)
    mat = mat.subtract(mean).divide(std)
    sample['feat'] = mat
    return sample


def sort_by_feats(sample):
    assert 'feat' in sample
    assert isinstance(sample['feat'], torch.Tensor)
    return sample['feat'].size(0)


def feats_length_fn(sample) -> int:
    assert 'feat' in sample
    return sample['feat'].size(0)


def compute_mfcc(sample,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, wav, feat, sample_rate, ...}
    """
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    mat = kaldi.mfcc(waveform,
                     num_mel_bins=num_mel_bins,
                     frame_length=frame_length,
                     frame_shift=frame_shift,
                     dither=dither,
                     num_ceps=num_ceps,
                     high_freq=high_freq,
                     low_freq=low_freq,
                     sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


def compute_log_mel_spectrogram(sample,
                                n_fft=400,
                                hop_length=160,
                                num_mel_bins=80,
                                padding=0,
                                pad_or_trim: bool = False,
                                max_duration: int = 30):
    """ Extract log mel spectrogram, modified from openai-whisper, see:
        - https://github.com/openai/whisper/blob/main/whisper/audio.py
        - https://github.com/wenet-e2e/wenet/pull/2141#issuecomment-1811765040

        Args:
            sample: {key, wav, sample_rate, ...}
            max_duration: valid when pad_or_trim is True (orign whisper style)

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav'].squeeze(0)  # (channel=1, sample) -> (sample,)
    if padding > 0:
        waveform = F.pad(waveform, (0, padding))
    if pad_or_trim:
        length = max_duration * sample_rate
        if waveform.size(0) >= length:
            waveform = waveform[:length]
        else:
            waveform = F.pad(waveform, (0, length - waveform.size(0)))

    window = torch.hann_window(n_fft)
    stft = torch.stft(waveform,
                      n_fft,
                      hop_length,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    sample['feat'] = log_spec.transpose(0, 1)
    return sample


def tokenize(sample, tokenizer: BaseTokenizer):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            sample: {key, wav, txt, sample_rate, ...}

        Returns:
            {key, wav, txt, tokens, label, sample_rate, ...}
    """
    assert 'txt' in sample
    tokens, label = tokenizer.tokenize(sample['txt'])
    sample['tokens'] = tokens
    sample['label'] = label
    return sample


def filter(sample,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            sample: {key, wav, label, sample_rate, ...}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            bool: True to keep, False to filter
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    # sample['wav'] is torch.Tensor, we have 100 frames every second
    num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False

    if 'label' in sample:
        if len(sample['label']) < token_min_length:
            return False
        if len(sample['label']) > token_max_length:
            return False
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                return False
            if len(sample['label']) / num_frames > max_output_input_ratio:
                return False
    return True


def spec_aug(sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            sample: {key, feat, ...}
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            {key, feat, ....}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    max_freq = y.size(1)
    # time mask
    for i in range(num_t_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for _ in range(num_f_mask):
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    sample['feat'] = y
    return sample


def spec_sub(sample, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation
        ref: U2++, section 3.2.3 [https://arxiv.org/abs/2106.05642]

        Args:
            sample: Iterable{key, feat, ...}
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            {key, feat, ...}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    y = x.clone().detach()
    max_frames = y.size(0)
    for _ in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = x[start - pos:end - pos, :]
    sample['feat'] = y
    return sample


def spec_trim(sample, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            sample: {key, feat, label}
            max_t: max width of length trimming

        Returns:
            {key, feat, label}
    """
    assert 'feat' in sample
    x = sample['feat']
    assert isinstance(x, torch.Tensor)
    max_frames = x.size(0)
    length = random.randint(1, max_t)
    if length < max_frames / 2:
        y = x.clone().detach()[:max_frames - length]
        sample['feat'] = y
    return sample


def padding(data):
    """ Padding the data into training data

        Args:
            data: List[{key, feat, label}

        Returns:
            Tuple(keys, feats, labels, feats lengths, label lengths)
    """
    sample = data
    assert isinstance(sample, list)
    feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                dtype=torch.int32)
    order = torch.argsort(feats_length, descending=True)
    feats_lengths = torch.tensor([sample[i]['feat'].size(0) for i in order],
                                 dtype=torch.int32)
    sorted_feats = [sample[i]['feat'] for i in order]
    sorted_keys = [sample[i]['key'] for i in order]
    sorted_labels = [
        torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
    ]
    sorted_wavs = [sample[i]['wav'].squeeze(0) for i in order]
    langs = [sample[i]['lang'] for i in order]
    tasks = [sample[i]['task'] for i in order]
    label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                 dtype=torch.int32)
    wav_lengths = torch.tensor([x.size(0) for x in sorted_wavs],
                               dtype=torch.int32)
    padded_feats = pad_sequence(sorted_feats,
                                batch_first=True,
                                padding_value=0)
    padding_labels = pad_sequence(sorted_labels,
                                  batch_first=True,
                                  padding_value=-1)
    padded_wavs = pad_sequence(sorted_wavs, batch_first=True, padding_value=0)

    batch = {
        "keys": sorted_keys,
        "feats": padded_feats,
        "target": padding_labels,
        "feats_lengths": feats_lengths,
        "target_lengths": label_lengths,
        "pcm": padded_wavs,
        "pcm_length": wav_lengths,
        "langs": langs,
        "tasks": tasks,
    }
    if 'speaker' in sample[0]:
        speaker = torch.tensor([sample[i]['speaker'] for i in order],
                               dtype=torch.int32)
        batch['speaker'] = speaker
    return batch


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False
