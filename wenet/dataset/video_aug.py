# Copyright (c) 2025 Fei Su (shinji721@outlook.com)
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

import cv2
import numpy as np
import math
import torch
import random


class Compose(object):
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize ndarray frames with mean and std (element-wise)."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        # frames: (T, H, W) or (T, H, W, C) as float/uint8
        return (frames - self.mean) / self.std

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'


class CenterCrop(object):
    def __init__(self, size):
        self.size = size  # (th, tw)

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw))/2.)
        delta_h = int(round((h - th))/2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames



class RandomCrop(object):
    """Crop the given image at the center
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w-tw)
        delta_h = random.randint(0, h-th)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class HorizontalFlip(object):
    """Flip image horizontally.
    """
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames


def advanced_resize(img, target_size, keep_ratio=False, pad_value=0):
    if keep_ratio: 
        origin_h, origin_w = img.shape[:2]
        target_w, target_h = target_size
        target_w_h_ratio = target_w / target_h
        img = img.copy()
        if (origin_w/origin_h) > target_w_h_ratio:
            # 此时原图过宽，需要pad上下两侧
            pad_size = (origin_w - target_w_h_ratio * origin_h) / target_w_h_ratio
            pad_half = int(pad_size/2)
            pad_data = np.ones((pad_half,origin_w,3), dtype=np.uint8) * int(pad_value)
            img = np.concatenate([pad_data,img,pad_data], axis=0)
        else:
            # 此时原图过窄，需要pad左右两侧
            pad_size = target_w_h_ratio * origin_h - origin_w 
            pad_half = int(pad_size/2)
            pad_data = np.ones((origin_h,pad_half,3), dtype=np.uint8) * int(pad_value)
            img = np.concatenate([pad_data, img,pad_data], axis=1)
            
    reshaped_img = cv2.resize(img, target_size)

    return reshaped_img


def load_video(path, time_clip=None, max_retries=3):
    last_err = None
    for i in range(max_retries):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))            

            if time_clip is not None:
                start_s, end_s = time_clip
                p1 = int(round(max(0.0, start_s) * fps))
                p2 = int(round(end_s * fps))
            else:
                p1, p2 = 0, total_frames

            p1 = max(0, p1)
            p2 = max(p1, p2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, p1)

            frames = []
            for _ in range(p1, min(p2, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            cap.release()
            if not frames:
                raise ValueError(f"No frames loaded from {path}")
            return np.stack(frames, axis=0), fps

        except Exception as e:
            print(f"failed loading {path} ({i+1}/{max_retries})--{e}")
            last_err = e

    raise ValueError(f"Unable to load {path}") from last_err  

def resize_video(frames, w, h, keep_ratio=False, pad_value=0):
    """
    在frames还为int(0-255)时,调整大小
    """
    if frames.ndim == 3:
        T, H, W = frames.shape
        if (H == h) and (W == w):
            return frames
        out = [advanced_resize(frames[t], (w, h), keep_ratio=keep_ratio, pad_value=pad_value)
               for t in range(T)]
        return np.stack(out, axis=0)

    elif frames.ndim == 4:
        T, H, W, C = frames.shape
        if (H == h) and (W == w):
            return frames
        out = [advanced_resize(frames[t], (w, h), keep_ratio=keep_ratio, pad_value=pad_value)
               for t in range(T)]
        return np.stack(out, axis=0)

    else:
        raise ValueError(f"resize_video expects (T,H,W) or (T,H,W,C), got {frames.shape}")


def augment_frames_numpy(
    frames,
    train=False,
    image_crop_size=88,
    image_mean=0.421,
    image_std=0.165
):
    if frames.ndim == 4:
        if frames.shape[-1] == 1:
            frames = np.squeeze(frames, axis=-1)
        else:
            raise ValueError(f"Expected frames input with 1 channel, got shape {frames.shape}")
    elif frames.ndim != 3:
        raise ValueError(f"Expected (T,H,W) or (T,H,W,C), got {frames.shape}")
    if train:
        transform = Compose([
            Normalize(0.0, 255.0),
            RandomCrop((image_crop_size, image_crop_size)),
            HorizontalFlip(0.5),
            Normalize(image_mean, image_std),    
        ])
    else:
        transform = Compose([
            Normalize(0.0, 255.0),
            CenterCrop((image_crop_size, image_crop_size)),
            Normalize(image_mean, image_std),
        ])
    proc = transform(frames)  # (T, H, W)
    proc = np.expand_dims(proc, axis=-1)   # (T, H, W, C)
    return np.ascontiguousarray(proc.astype(np.float32))


def augment_pt_video(
    video_tensor: torch.Tensor,
    train: bool = False,
    image_crop_size: int = 88,
    image_mean: float = 0.421,
    image_std: float = 0.165
) -> torch.Tensor:
    """
    Apply augmentation to a loaded .pt tensor.

    Assumes input is [T, H, W, 3] float32, values in [0.0, 1.0].
    Always returns [T, h, w, 1] float32.
    """
    # convert to numpy grayscale first
    # import ipdb; ipdb.set_trace()
    arr = video_tensor.cpu().numpy()  # (T,96,96,3), float32 in [0,1]
    B, G, R = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * R + 0.587 * G + 0.114 * B  # (T,H,W)

    # build transforms
    if train:
        transform = Compose([
            RandomCrop((image_crop_size, image_crop_size)),
            HorizontalFlip(0.5),
            Normalize(image_mean, image_std),
        ])
    else:
        transform = Compose([
            CenterCrop((image_crop_size, image_crop_size)),
            Normalize(image_mean, image_std),
        ])

    proc = transform(gray)  # (T,h,w)

    proc = np.expand_dims(proc, axis=-1)       # (T, 88, 88, 1)
    # proc = np.repeat(proc, 3, axis=-1)         # (T, 88, 88, 3)
    proc = np.ascontiguousarray(proc)
    # import ipdb; ipdb.set_trace()
    return torch.from_numpy(proc).to(dtype=torch.float32, device=video_tensor.device)
