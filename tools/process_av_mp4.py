import os
import glob
import subprocess
import argparse
from tqdm import tqdm
from multiprocessing import Pool

def write_scp(mp4_files, audio_dir, video_dir, wav_scp_path, video_scp_path):
    with open(wav_scp_path, "w", encoding="utf-8") as wav_f, \
         open(video_scp_path, "w", encoding="utf-8") as video_f:
        for mp4 in mp4_files:
            utt_id = os.path.splitext(os.path.basename(mp4))[0]
            audio_path = os.path.join(audio_dir, f"{utt_id}_audio.wav")
            video_path = os.path.join(video_dir, f"{utt_id}_video.mp4")
            wav_f.write(f"{utt_id} {audio_path}\n")
            video_f.write(f"{utt_id} {video_path}\n")
    print(f"wav.scp saved to {wav_scp_path}")
    print(f"video.scp saved to {video_scp_path}")

def process_mp4(args):
    mp4_path, audio_out_dir, video_out_dir = args
    try:
        base_name = os.path.splitext(os.path.basename(mp4_path))[0]
        video_out = os.path.join(video_out_dir, base_name + "_video.mp4")
        audio_out = os.path.join(audio_out_dir, base_name + "_audio.wav")

        os.makedirs(video_out_dir, exist_ok=True)
        os.makedirs(audio_out_dir, exist_ok=True)

        # Extract raw video
        video_cmd = [
            "ffmpeg", "-i", mp4_path, "-an",
            "-vf", "scale=1280:720", "-r", "25",
            "-c:v", "libx264", "-y", video_out
        ]

        # Extract raw audio and convert to mono, 16kHz
        audio_cmd = [
            "ffmpeg", "-i", mp4_path, "-vn",
            "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            "-y", audio_out
        ]

        subprocess.run(video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {mp4_path}: {e}")
        return False

def multiprocess_av_mp4(av_data_dir, out_dir, num_workers=32):
    mp4_files = glob.glob(os.path.join(av_data_dir, "*.mp4"))
    mp4_files.sort()

    if not mp4_files:
        print("No MP4 files found.")
        return

    audio_out_dir = os.path.join(out_dir, "audio_data")
    video_out_dir = os.path.join(out_dir, "video_data")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)

    print("Start extracting video and audio ...")
    args_list = [(mp4, audio_out_dir, video_out_dir) for mp4 in mp4_files]

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_mp4, args_list), total=len(mp4_files)))

    # Write scp files
    # write_scp(
    #     mp4_files,
    #     audio_out_dir,
    #     video_out_dir,
    #     os.path.join(out_dir, "wav.scp"),
    #     os.path.join(out_dir, "video.scp")
    # )

    print("All done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--av_data", type=str, required=True, help="Path to input AV .mp4 files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for extracted files and scp")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel processes")
    args = parser.parse_args()

    multiprocess_av_mp4(args.av_data, args.out_dir, args.num_workers)
