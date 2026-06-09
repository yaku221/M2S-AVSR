import os
import json
import wave
import argparse
import cv2
from pathlib import Path
from collections import defaultdict


def get_wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def get_mp4_duration(mp4_path: Path) -> float:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def process_directory(input_dir, output_dir, suffixes, scp_name, use_duration=False, min_duration=0.0):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Debug] input_dir = {input_dir}")
    print(f"[Debug] output_dir = {output_dir}")
    print(f"[Debug] suffixes = {suffixes}")
    print(f"[Debug] files found = {len(list(Path(input_dir).rglob('*')))}")
    
    data_list = []
    text_entries = []
    scp_entries = []
    duration_tracker = defaultdict(float)
    total_duration = 0.0

    for file_path in Path(input_dir).rglob("*"):
        print(f"[Debug] checking file: {file_path}")
        
        if file_path.suffix.lower() not in suffixes:
            continue

        key = file_path.stem
        text = "占位符"

        if file_path.suffix.lower() == ".wav":
            duration = get_wav_duration(file_path)
        elif file_path.suffix.lower() in {".mp4", ".avi"}:
            duration = get_mp4_duration(file_path)
        else:
            duration = 0.0

        if duration < min_duration:
            print(f"[Skip] {key}: duration {duration:.2f}s < {min_duration:.2f}s")
            continue

        data_list.append({
            "key": key,
            "wav": str(file_path.resolve()),
            "txt": text
        })

        text_entries.append(f"{key} {text}")
        if use_duration:
            scp_entries.append(f"{key} {file_path} {duration:.2f}")
            prefix = "_".join(key.split("_")[:3])
            duration_tracker[prefix] += duration
            total_duration += duration
        else:
            scp_entries.append(f"{key} {file_path}")

    with open(os.path.join(output_dir, "data.list"), 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(os.path.join(output_dir, "text"), 'w', encoding='utf-8') as f:
        f.write("\n".join(text_entries) + "\n")

    with open(os.path.join(output_dir, scp_name), 'w', encoding='utf-8') as f:
        f.write("\n".join(scp_entries) + "\n")

    print(f"[{scp_name}] Saved {len(data_list)} entries to {output_dir}")

    if use_duration and len(duration_tracker) > 0:
        print(f"=== Duration by prefix in {scp_name} ===")
        for prefix, total_dur in sorted(duration_tracker.items()):
            print(f"{prefix}: {total_dur:.2f} second")
        print(f"=== Total Duration in {scp_name} ===\nTotal: {total_duration:.2f} seconds")


def prepare_data(wav_dir, lips_dir, output_dir, use_duration=False, min_duration=0.0):
    audio_out = os.path.join(output_dir, "audio")
    video_out = os.path.join(output_dir, "video")

    process_directory(wav_dir, audio_out, {".wav"}, "wav.scp", use_duration, min_duration)
    process_directory(lips_dir, video_out, {".mp4", ".avi"}, "video.scp", use_duration, min_duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', type=str, required=True, help='Directory containing .wav files')
    parser.add_argument('--lips_data_dir', type=str, required=True, help='Directory containing lip videos (.mp4/.avi)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save audio/video metadata')
    parser.add_argument('--duration', action='store_true', help='Compute and include durations in scp files')
    parser.add_argument('--min_duration', type=float, default=0.0, help='Minimum duration to keep (in seconds)')
    args = parser.parse_args()

    prepare_data(args.wav_dir, args.lips_data_dir, args.output_dir, use_duration=args.duration, min_duration=args.min_duration)
