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

import re
from collections import defaultdict
from typing import Tuple, Optional, Dict
import json


__all__ = [
    "build_multiview_index",
    "parse_realav1_data", 
    "parse_misp2021_data",
    "parse_general_data",    
]


def build_multiview_index(video_data_list_file: str, dataset_name: str) -> Dict[str, Dict[str, str]]:
    key2views: Dict[str, Dict[str, str]] = defaultdict(dict)
    pat = re.compile(r'_(D[012])_')   # _D0_/_D1_/_D2_
    with open(video_data_list_file, 'r', encoding='utf-8') as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                vkey = obj.get('key')
                vpath = obj.get('wav')
                if not vkey or not vpath:
                    continue
            except Exception:
                continue
            if dataset_name == "realav1":
                m = pat.search(vkey)
                if not m:                    
                    continue
                view = m.group(1)  # D0/D1/D2
                base_key = pat.sub('_', vkey, count=1)  # del _D{n}_ ==> base key
                key2views[base_key][view] = vpath
            else:                
                # other datasets
                base_key = vkey
                key2views[base_key]["D1"] = vpath
    return dict(key2views)



def _strip_ext(s: str) -> str:
    for ext in (".flac", ".wav", ".mp3", ".avi", ".mp4", ".mkv", ".mov", ".pt", ".pkl"):
        if s.lower().endswith(ext):
            return s[: -len(ext)]
    return s



# ---------- for realav1 ----------
def _count_speakers(s_ids: str) -> int:
    if not (isinstance(s_ids, str) and s_ids.startswith('S')):
        return 0
    digits = "".join(ch for ch in s_ids[1:] if ch.isdigit())
    if not digits:
        return 0
    return len(digits) // 3


def parse_realav1_data(key: str, max_speakers: Optional[int] = None) -> Tuple[str, Optional[float], Optional[float]]:
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")

    raw = key
    key = _strip_ext(key)
    
    # for realav1 raw data
    if key.startswith("S"):
        # key: S001_L1_S001005006_G41_P5_Near_001_001580-001700   
        # key: <Sspkid>_<location_id>_<spks>_<group_id)>_<permutation_id>_<field>_<spkid>_<start>-<end>
        #       0             1         2       3                4          5        6        -1
        # spk_session_id: S001_L1_S001005006_G41_P5_Near   
        # key = sample['key']
        parts = key.split("_")  # S001_L1_S001005006_G41_P5_Near_001_001580-001700
        spkid = parts[0]  # S000
        session_id_prefix = "_".join(parts[1:5])  # L1_S001005006_G41_P5
        audio_field = parts[5]  # Near
        if len(parts) >= 3 and parts[2].startswith("S") and max_speakers is not None:
                n_spk = _count_speakers(parts[2])
                if n_spk > max_speakers:
                    # print(f"max_speakers is {max_speakers}, ignore {session_id_prefix}")
                    return None, None, None       
        
        start = int(parts[-1].split('-')[0]) / 100  # 15.80
        end = int(parts[-1].split('-')[1]) / 100  # 17.00
        start_timestamp = parts[-1].split('-')[0]
        end_timestamp = parts[-1].split('-')[1]    

        # S001_L1_S001005006_G41_P5_Near_001_001580-001700 ==> 
        # S001_L1_S001005006_G41_P5_D0_LIP_001580-001700.avi
        # S001_L1_S001005006_G41_P5_D1_LIP_001580-001700.avi
        # S001_L1_S001005006_G41_P5_D2_LIP_001580-001700.avi
        spk_session_id = f"{spkid}_{session_id_prefix}_LIP_{start_timestamp}-{end_timestamp}"
        # realav1 raw audio data has been segmented, no need start/end
        return spk_session_id, None, None

    # for realav1 gss data
    if key.startswith("L"):
        # key: L3_S138136137_G37_P5_Far-136-036356_036584
        
        # realav1 audio:
        # L3_S138136137_G37_P5_Far-136-036356_036584       
        
        # realav1 video:
        # S136_L3_S138136137_G37_P5_D0_LIP_036356-036584.avi
        # S136_L3_S138136137_G37_P5_D1_LIP_036356-036584.avi
        # S136_L3_S138136137_G37_P5_D2_LIP_036356-036584.avi       
        
        parts = key.split("-")  # L3_S138136137_G37_P5_Far-136-036356_036584     
        session_id = parts[0]  # L3_S138136137_G37_P5_Far
        session_id_prefix = "_".join(session_id.split("_")[:-1])  # L3_S138136137_G37_P5
        spkid = 'S' + parts[1]  # S136
        tail = parts[-1]  # 036356_036584
        audio_field = session_id.split("_")[-1]
        _segments = session_id.split("_")
        # print(f"max_speakers is {max_speakers}, ignore {session_id_prefix}")
        if len(_segments) >= 2 and _segments[1].startswith("S") and max_speakers is not None:
            n_spk = _count_speakers(_segments[1])
            if n_spk > max_speakers:
                # print(f"max_speakers is {max_speakers}, ignore {session_id_prefix}")
                return None, None, None

        start = int(tail.split('_')[0]) / 100  # 363.56
        end = int(tail.split('_')[1]) / 100  # 365.84
        start_timestamp = tail.split('_')[0]
        end_timestamp = tail.split('_')[1]
        
        # S136_L3_S138136137_G37_P5_LIP_036356-036584
        spk_session_id = f"{spkid}_{session_id_prefix}_LIP_{start_timestamp}-{end_timestamp}"
        # realav1 raw audio data has been segmented, no need start/end
        return spk_session_id, None, None


# ---------- for misp2021 ----------
def _digits_only(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def parse_misp2021_data(key: str) -> Tuple[str, Optional[float], Optional[float]]:
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")

    raw = key
    key = _strip_ext(key)
    
    # for misp2021 raw data
    if key.startswith("S"):
        # key: S000_R01_S000001_C07_I0_Far_0_003376-003784   S409_R70_S409410411_C03_I0_Near_409_120196-120432
        # key: <spkid>_<roomid>_<spks>_<configid>_<index>_<field>_..._<start>-<end>
        #       0        1        2       3          4       5             -1
        # spk_session_id: S000_R01_S000001_C07_I0_Middle   S000_R01_S000001_C07_I0_Far   
        # key = sample['key']
        parts = key.split("_")  # S409_R70_S409410411_C03_I0_Near_409_120196-120432
        spkid = parts[0]  # S000
        session_id_prefix = "_".join(parts[1:5])  # R01_S000001_C07_I0
        audio_field = parts[5]  # Near
        
        start = int(parts[-1].split('-')[0]) / 100  # 4.12
        end = int(parts[-1].split('-')[1]) / 100  # 4.80
        start_timestamp = parts[-1].split('-')[0]
        end_timestamp = parts[-1].split('-')[1]    

        lip_field = "middle" if audio_field == "Near" else "far"
        spk_session_id = f"{spkid}_{session_id_prefix}_LIP_{lip_field}_{start_timestamp}-{end_timestamp}"
        return spk_session_id, start, end

    # for misp2021 gss data
    if key.startswith("R"):
        # key: R07_S151152153_C02_I0_Middle-153-068532_069140.flac
        # key: R01_S000001_C07_I0_Far-000-002624_002748.flac
        
        # misp2021 audio:
        # R78_S446447448_C06_I0_Far-446-000476_000548 
        # R01_S041042043_C02_I0_Middle-041-120524_120784   R67_S395396397398_C03_I1_Middle-398-083716_084172
        
        # misp2021 video:
        # S446_R78_S446447448_C06_I0_LIP_middle_000476-000548
        # S446_R78_S446447448_C06_I0_LIP_far_000476-000548    S398_R67_S395396397398_C03_I1_LIP_middle_083716-084172
        
        # misp2022 audio:
        # 2022 eval_far gsskey: R69_S405406_C07_I1-1-027207_027246.flac ==>
        # 2022 eval_far lipskey: S0_R01_S016017_C02_I1_Far
        # R52_S280281282_C03_I0-0-015734_015834 R52_S280281282_C03_I0-0-015734 S0
        parts = key.split("-")  # R01_S000001_C07_I0_Far-000-002624_002748        
        session_id = parts[0]  # R01_S000001_C07_I0_Far
        spkid = 'S' + parts[1]  # S000
        tail = parts[-1]  # 002624_002748
        audio_field = session_id.split("_")[-1]

        start = int(tail.split('_')[0]) / 100  # 4.12
        end = int(tail.split('_')[1]) / 100  # 4.80
        start_timestamp = tail.split('_')[0]
        end_timestamp = tail.split('_')[1]

        lip_field = "middle" if audio_field == "Near" else "far"
        session_id_prefix = "_".join(session_id.split("_")[:-1])  # R01_S000001_C07_I0
        
        # S446_R78_S446447448_C06_I0_LIP_far_000476-000548 
        spk_session_id = f"{spkid}_{session_id_prefix}_LIP_{lip_field}_{start_timestamp}-{end_timestamp}"
        return spk_session_id, start, end



# ---------- for general data ----------
def parse_general_data(key: str) -> Tuple[str, Optional[float], Optional[float]]:
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")

    spk_session_id = key
    # for dining room data
    if "audio" in spk_session_id:
        spk_session_id = spk_session_id.replace("audio", "video")

    # for submission
    if "wav" in spk_session_id:
        spk_session_id = spk_session_id.replace("wav", "lip")   

    return spk_session_id, None, None
