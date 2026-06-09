# encoding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import argparse
import multiprocessing as mp
from math import ceil
from random import shuffle

from deeplab.toolkit.RetinaFace.api import FaceDetection
from deeplab.utils.fileio import init_output_dir, save_json
from deeplab.utils.pbar import ProgressBar

def run(tasks, worker, output_dir, batch_size, q1, q2):
    torch.cuda.set_device(worker)
    face_detector = FaceDetection(device=worker)
    for task in tasks:
        reco = task['reco']    # 3m_001_video
        path = task['path']    # .../3m_001_video.mp4
        # local_max_face=(len(path.split('/')[-1].split('.')[0].split('_')[1])-1) // 3  # for misp2021
        local_max_face = 1     # for dining room
        # print(f"processing {reco} on GPU:{worker}, local_max_face: {local_max_face}")
        print(f"processing {reco} on GPU:{worker}")
        if not os.path.exists(path):
            q2.put(task)
        else:
            try:
                dets_dict = face_detector.predict_video(
                    video_path=path,
                    max_face=local_max_face,
                    batch_size=batch_size
                )
                save_path = os.path.join(output_dir, reco + '.json')
                init_output_dir(save_path)
                save_json(save_path, dets_dict)
                q1.put(1)
            except Exception as e:
                convert = dict()
                convert['error'] = str(e)
                q2.put(convert)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face detection on videos.")
    parser.add_argument('--video_dir', required=True, help="Directory containing video files.")
    parser.add_argument('--output_dir', required=True, help="Directory to save face detection results.")
    parser.add_argument('--gpu_num', type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--nj", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    mp.set_start_method("spawn")
    batch_size = args.nj
    num_workers_per_device = 1

    devices = [f'cuda:{i}' for i in range(args.gpu_num)]

    video_dir = args.video_dir
    output_dir = args.output_dir

    tasks = []
    for vfile in os.listdir(video_dir):
        path = os.path.join(video_dir, vfile)
        if path.endswith('.mp4'):
            reco = os.path.splitext(vfile)[0]   # 3m_001_video
            tasks.append(dict(reco=reco, path=path))

    shuffle(tasks)
    pbar = ProgressBar(total=len(tasks))

    workers = []
    for _ in range(num_workers_per_device):
        workers += devices

    interval = ceil(len(tasks) / len(workers))
    async_result = []
    process_pool = mp.Pool(len(workers))

    for i, w in enumerate(workers):
        sub_tasks = tasks[i*interval : i*interval + interval]
        res = process_pool.apply_async(
            run,
            args=(sub_tasks, w, output_dir, batch_size, pbar.count_queue, pbar.error_queue)
        )
        async_result.append(res)

    process_pool.close()
    process_pool.join()

    for res in async_result:
        res.get()

    pbar.close()
    pbar.save_error_to_json(os.path.join(output_dir, 'errors.json'))
