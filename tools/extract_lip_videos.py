# encoding: utf-8

import os
import sys
import argparse
import cv2
import multiprocessing as mp
from math import ceil
from random import shuffle
import numpy as np
from copy import deepcopy
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deeplab.utils.pbar import ProgressBar
from deeplab.utils.fileio import init_output_dir, read_json, save_json, load_rttm
from deeplab.data.image import crop_squared_image
from deeplab.toolkit.ArcFace.face_alignment import align_and_crop_face
from deeplab.metric.distance import get_euclidean_distance


def get_lip_bbox(face_5pts):
    # face_5pts: [x_eye_l, y_eye_l, x_eye_r, y_eye_r, x_nose, y_nose, x_mouth_l, y_mouth_l, x_mouth_r, y_mouth_r]
    mouth_x1 = face_5pts[6]
    mouth_y1 = face_5pts[7]
    mouth_x2 = face_5pts[8]
    mouth_y2 = face_5pts[9]
    cx = (mouth_x1 + mouth_x2) / 2
    cy = (mouth_y1 + mouth_y2) / 2
    d_mn = abs(face_5pts[5] - cy)
    width = min(3.2 * d_mn, max(2.0 * d_mn, 2.0 * (mouth_x2 - mouth_x1)))
    lip_x1 = int(cx - width / 2)
    lip_y1 = int(cy - width / 2)
    lip_x2 = int(cx + width / 2)
    lip_y2 = int(cy + width / 2)
    return [lip_x1, lip_y1, lip_x2, lip_y2]


def get_reco2spks_infar(reco):
    # Example: reco="M027_S197199200201241242243_F8N_Far"
    # Extract 3-digit speaker IDs from the second part
    reco2spks = {}
    # The part after first underscore: e.g. "S197199200201241242243"
    # user logic: slice them in steps of 3
    parts = reco.split("_", maxsplit=1)
    if len(parts) < 2:
        reco2spks[reco] = []
        return reco2spks

    spk_str = parts[1]
    speaker_list = [spk_str[i:i+3] for i in range(1, len(spk_str), 3)]
    reco2spks[reco] = speaker_list
    return reco2spks


def face_clustering_for_misp(dets_dict):
    """
    Cluster face bounding boxes to group them into different face_id.
    """
    meta_list = []
    for frame_idx, dets in deepcopy(dets_dict).items():
        for meta in dets:
            meta.update(dict(frame_idx=frame_idx))
            meta_list.append(meta)

    # number of clusters = median #faces per frame
    num_cluster = int(np.median([len(dets) for dets in dets_dict.values()]))

    bbox_matrix = np.array([meta['face_bbox'] for meta in meta_list])
    labels = KMeans(n_clusters=num_cluster, random_state=0).fit(bbox_matrix).labels_
    labels = [str(label) for label in labels]

    label2bbox = {}
    for i, label in enumerate(labels):
        if label not in label2bbox:
            label2bbox[label] = []
        label2bbox[label].append(bbox_matrix[i])

    active_labels = []
    for label, bbox_list in label2bbox.items():
        bbox_std = np.array(bbox_list).std(axis=0).mean()
        if bbox_std > 1:
            active_labels.append(label)

    for i, label in enumerate(labels):
        if label in active_labels:
            meta_list[i]['face_id'] = label

    clus_dets_dict = {}
    for meta in meta_list:
        frame_idx = meta['frame_idx']
        if 'face_id' not in meta:
            continue
        if frame_idx not in clus_dets_dict:
            clus_dets_dict[frame_idx] = []
        clus_dets_dict[frame_idx].append(meta)

    label2cluster_center = {}
    for label, bbox_list in label2bbox.items():
        if label in active_labels:
            label2cluster_center[label] = np.array(bbox_list).mean(axis=0)

    # for each frame, keep only the bounding box that is closest to the cluster center if there's multiple
    for frame_idx, dets in clus_dets_dict.items():
        label2dist = {}
        for meta in dets:
            face_id = meta['face_id']
            if face_id not in label2dist:
                label2dist[face_id] = []
            center_bbox = label2cluster_center[face_id]
            center_dist = get_euclidean_distance(center_bbox, meta['face_bbox'])
            label2dist[face_id].append(center_dist)

        new_dets = []
        for meta in dets:
            face_id = meta['face_id']
            center_bbox = label2cluster_center[face_id]
            center_dist = get_euclidean_distance(center_bbox, meta['face_bbox'])
            if center_dist == min(label2dist[face_id]):
                new_dets.append(meta)

        clus_dets_dict[frame_idx] = new_dets

    return clus_dets_dict, label2cluster_center


def match_to_label(reco, gt_det_path, rttm_path, dets_dict, label2cluster_center):
    """
    Match face_id (cluster label) to speaker ID from ground truth data.
    """
    gt = read_json(gt_det_path)
    reco2spks = get_reco2spks_infar(reco)

    gt_spk2center = {}
    for metas in gt.values():
        for meta in metas:
            spk_id = str(meta['id']).zfill(3)
            if spk_id not in reco2spks[reco]:
                continue
            if spk_id not in gt_spk2center:
                gt_spk2center[spk_id] = []
            # average face center
            gt_spk2center[spk_id].append(
                [(meta['x1'] + meta['x2']) / 2, (meta['y1'] + meta['y2']) / 2]
            )

    for k, v in gt_spk2center.items():
        gt_spk2center[k] = np.array(v).mean(axis=0)

    src_ids = list(label2cluster_center.keys())
    dst_ids = list(gt_spk2center.keys())

    cost_matrix = np.zeros((len(src_ids), len(dst_ids)))
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            x1, y1, x2, y2 = label2cluster_center[src_ids[i]]
            cost_matrix[i, j] = get_euclidean_distance(
                [(x1 + x2) / 2, (y1 + y2) / 2],
                gt_spk2center[dst_ids[j]]
            )

    mapping = {}
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, j in zip(row_ind, col_ind):
        mapping[src_ids[i]] = dst_ids[j]

    mapped_dets_dict = {}
    for frame_idx, dets in dets_dict.items():
        mapped_dets = []
        for meta in dets:
            if meta['face_id'] in mapping:
                meta['face_id'] = mapping[meta['face_id']]
                mapped_dets.append(meta)
        if len(mapped_dets) > 0:
            mapped_dets_dict[frame_idx] = mapped_dets

    return mapped_dets_dict


def extract_face_videos(video_path, dets_dict, output_dir, reco, crop_type, crop_size, filter_size=0):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open video file: {video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    spk2dets_dict = {}
    for frame_idx, dets in dets_dict.items():
        for meta in dets:
            face_id = meta['face_id']
            if face_id not in spk2dets_dict:
                spk2dets_dict[face_id] = {}
            spk2dets_dict[face_id][frame_idx] = meta

    if filter_size > 0:
        for spk_id, sub_dets_dict in spk2dets_dict.items():
            new_dets_dict = deepcopy(sub_dets_dict)
            for frame_idx, meta in sub_dets_dict.items():
                f1 = int(int(frame_idx) - filter_size / 2)
                f2 = int(int(frame_idx) + filter_size / 2)
                bbox_window = []
                for f_index in range(f1, f2):
                    if str(f_index) in sub_dets_dict:
                        bbox_window.append(sub_dets_dict[str(f_index)]['face_bbox'])
                if len(bbox_window) > 0:
                    bbox_median = np.median(bbox_window, axis=0).astype('int').tolist()
                    new_dets_dict[frame_idx]['face_bbox'] = bbox_median
            spk2dets_dict[spk_id] = new_dets_dict

    spk2video_writer = {}
    for spk_id in spk2dets_dict.keys():
        dst_path = os.path.join(output_dir, f'{reco}_{spk_id}.mp4')
        init_output_dir(dst_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        spk2video_writer[spk_id] = cv2.VideoWriter(dst_path, fourcc, fps, (crop_size, crop_size), True)

    for f_idx in range(total_frames):
        ret, img = cap.read()
        if not ret:
            for video_writer in spk2video_writer.values():
                padded_face = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                video_writer.write(padded_face)
        else:
            frame_str = str(f_idx)
            for spk_id, video_writer in spk2video_writer.items():
                if frame_str not in spk2dets_dict[spk_id]:
                    padded_face = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    video_writer.write(padded_face)
                else:
                    meta = spk2dets_dict[spk_id][frame_str]
                    if crop_type == 'aligned':
                        face = align_and_crop_face(img, meta['face_5pts'], (crop_size, crop_size))
                    elif crop_type == 'squared':
                        face = crop_squared_image(img, meta['face_bbox'], (crop_size, crop_size))
                    else:
                        raise NotImplementedError('Invalid face crop type.')
                    video_writer.write(face)

    cap.release()
    for writer in spk2video_writer.values():
        writer.release()


def extract_lip_videos(video_path, dets_dict, output_dir, reco, crop_size, filter_size=0):
    for frame_idx, dets in dets_dict.items():
        for meta in dets:
            meta['lip_bbox'] = get_lip_bbox(meta['face_5pts'])

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open video file: {video_path}'

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    spk2dets_dict = {}
    for frame_idx, dets in dets_dict.items():
        for meta in dets:
            face_id = meta['face_id']
            if face_id not in spk2dets_dict:
                spk2dets_dict[face_id] = {}
            spk2dets_dict[face_id][frame_idx] = meta

    if filter_size > 0:
        for spk_id, sub_dets_dict in spk2dets_dict.items():
            new_dets_dict = deepcopy(sub_dets_dict)
            for frame_idx, meta in sub_dets_dict.items():
                f1 = int(int(frame_idx) - filter_size / 2)
                f2 = int(int(frame_idx) + filter_size / 2)
                bbox_window = []
                for f_index in range(f1, f2):
                    if str(f_index) in sub_dets_dict:
                        bbox_window.append(sub_dets_dict[str(f_index)]['lip_bbox'])
                if len(bbox_window) > 0:
                    bbox_median = np.median(bbox_window, axis=0).astype('int').tolist()
                    new_dets_dict[frame_idx]['lip_bbox'] = bbox_median
            spk2dets_dict[spk_id] = new_dets_dict

    spk2video_writer = {}
    for spk_id in spk2dets_dict.keys():
        dst_path = os.path.join(output_dir, f'{reco}.mp4')
        init_output_dir(dst_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        spk2video_writer[spk_id] = cv2.VideoWriter(dst_path, fourcc, fps, (crop_size, crop_size), True)

    for f_idx in range(total_frames):
        ret, img = cap.read()
        if not ret:
            for video_writer in spk2video_writer.values():
                padded_lip = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                video_writer.write(padded_lip)
        else:
            frame_str = str(f_idx)
            for spk_id, video_writer in spk2video_writer.items():
                if frame_str not in spk2dets_dict[spk_id]:
                    padded_lip = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    video_writer.write(padded_lip)
                else:
                    meta = spk2dets_dict[spk_id][frame_str]
                    lip = crop_squared_image(img, meta['lip_bbox'], (crop_size, crop_size))
                    video_writer.write(lip)

    cap.release()
    for writer in spk2video_writer.values():
        writer.release()


def run(tasks, rttm_path, dets_output_dir, face_output_dir, lip_output_dir, gt_dir, use_gt, q1, q2):
    for task in tasks:
        reco = task['reco']
        vid_path = task['vid_path']
        det_path = task['det_path']
        gt_det_path = task['gt_det_path']

        if not os.path.exists(vid_path):
            q2.put(task)
            continue

        try:
            dets_dict = read_json(det_path)
            dets_dict, label2cluster_center = face_clustering_for_misp(dets_dict)
            if use_gt:
                dets_dict = match_to_label(reco, gt_det_path, rttm_path, dets_dict, label2cluster_center)

            extract_face_videos(
                vid_path, dets_dict, face_output_dir + '_squared',
                reco, crop_type='squared', crop_size=112, filter_size=50
            )
            extract_lip_videos(
                vid_path, dets_dict, lip_output_dir,
                reco, crop_size=112, filter_size=50
            )

            save_json(os.path.join(dets_output_dir, reco + '.json'), dets_dict)
            q1.put(1)
        except:
            q2.put(task)


if __name__ == '__main__':
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing mp4 videos.')
    parser.add_argument('--rttm_path', type=str, default=None, help='Path to RTTM file if needed.')
    parser.add_argument('--gt_dir', type=str, default=None, help='Directory containing ground-truth JSON for matching.')
    parser.add_argument('--dets_result_dir', type=str, required=True, help='Directory of face detection JSON.')
    parser.add_argument('--dets_output_dir', type=str, required=True, help='Directory to save matched detection JSON.')
    parser.add_argument('--face_output_dir', type=str, required=True, help='Directory to save extracted face videos.')
    parser.add_argument('--lip_output_dir', type=str, required=True, help='Directory to save extracted lip videos.')
    parser.add_argument('--nj', type=int, default=8, help='Number of parallel processes.')
    parser.add_argument('--use_gt', action='store_true', help='Whether to match face_id with ground truth speaker ID.')
    args = parser.parse_args()

    video_dir = args.video_dir
    rttm_path = args.rttm_path
    gt_dir = args.gt_dir
    dets_result_dir = args.dets_result_dir
    dets_output_dir = args.dets_output_dir
    face_output_dir = args.face_output_dir
    lip_output_dir = args.lip_output_dir
    num_workers = args.nj
    use_gt = args.use_gt

    tasks = []
    for vfile in os.listdir(video_dir):
        if vfile.endswith('.mp4'):
            reco = os.path.splitext(vfile)[0]
            vid_path = os.path.join(video_dir, vfile)
            det_path = os.path.join(dets_result_dir, reco + '.json')
            if gt_dir is not None:
                gt_det_path = os.path.join(gt_dir, reco + '.json')
            else:
                gt_det_path = ""
            tasks.append(dict(
                reco=reco,
                vid_path=vid_path,
                det_path=det_path,
                gt_det_path=gt_det_path
            ))

    shuffle(tasks)
    pbar = ProgressBar(total=len(tasks))

    interval = ceil(len(tasks) / num_workers)
    process_pool = mp.Pool(num_workers)
    async_result = []

    for i in range(num_workers):
        sub_tasks = tasks[i * interval: i * interval + interval]
        res = process_pool.apply_async(
            run,
            args=(sub_tasks, rttm_path, dets_output_dir, face_output_dir, lip_output_dir, gt_dir, use_gt, pbar.count_queue, pbar.error_queue)
        )
        async_result.append(res)

    process_pool.close()
    process_pool.join()

    for res in async_result:
        res.get()

    pbar.close()
    pbar.save_error_to_json(os.path.join(dets_output_dir, 'errors.json'))
