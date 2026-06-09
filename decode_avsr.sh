#!/usr/bin/env bash

. ./path.sh || exit 1;

conda activate m2s_avsr

data_type=raw

work_path=/path/to/m2s_avsr

## func
_die () {
  echo "[ERROR] $*" >&2
  exit 2
}

# Select dataset
dataset_name=        # aishell8_rs/misp2021/lrs3


if [ "${dataset_name}" = "aishell8_rs" ]; then
  data_path="${work_path}/data/aishell8_rs"
  test_audio_set="${data_path}/audio/eval_far"
  test_video_set="${data_path}/video/eval"
  gt_text="${test_audio_set}/text"


elif [ "${dataset_name}" = "misp2021" ]; then
  data_path="${work_path}/data/misp2021"
  test_audio_set="${data_path}/audio/eval_far"
  test_video_set="${data_path}/video/eval_far"
  gt_text="${test_audio_set}/text"


elif [ "${dataset_name}" = "lrs3" ]; then
  data_path="${work_path}/data/lrs3"
  test_audio_set="${data_path}/audio/eval"
  test_video_set="${data_path}/video/eval"
  gt_text="${test_audio_set}/text"

else
  _die "Unknown dataset_name='${dataset_name}'. "
fi

echo "Dataset: ${dataset_name}"
echo "  data_path      = ${data_path}"
echo "  test_audio_set = ${test_audio_set}"
echo "  test_video_set = ${test_video_set}"
echo "  gt_text        = ${gt_text}"


dir=exp
mkdir -p $dir
echo "Exp dir: ${dir}"


decode_checkpoints=checkpoints/avsr.pt


# Decode config
decode_modes="attention"
decoding_chunk_size=-1
ctc_weight=0.0
reverse_weight=0.0
decode_batch=1


python wenet/bin/recognize_avsr.py \
  --gpu 0 --device "cuda" \
  --modes "${decode_modes}" \
  --config "checkpoints/config.yaml" \
  --data_type "${data_type}" \
  --test_audio_data "${test_audio_set}/data.list" \
  --test_video_data "${test_video_set}/data.list" \
  --checkpoint "${decode_checkpoint}" \
  --beam_size 25 \
  --batch_size "${decode_batch}" \
  --blank_penalty 0.0 \
  --ctc_weight "${ctc_weight}" \
  --reverse_weight "${reverse_weight}" \
  --result_dir "${dir}" \
  ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}


python tools/compute-wer.py --cs=1 --v=1 \
        "${gt_text}" "${dir}/${decode_modes}/text" > "${dir}/${decode_modes}/wer"
