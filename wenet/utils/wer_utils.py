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
import unicodedata
from typing import List, Tuple, Dict

_PUNCTS = set([
    '!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』', '《', '》'
])
_ASCII_PUNCTS = set('.,;:!?\'"()[]{}<>/\\|-_~`@#$%^&*+=…')
_PUNCTS = _PUNCTS | _ASCII_PUNCTS 
_SPACE_CHARS = set([' ', '\t', '\r', '\n'])
_TAG_RE = re.compile(r"<[^>]*>")  # remove tags like <unk> <noise> ...


def load_references(ref_path: str) -> Dict[str, str]:
    """Load reference file: each line = '<uttid> <reference text...>'"""
    refs: Dict[str, str] = {}
    with open(ref_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            uttid = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            refs[uttid] = text
    return refs


def _strip_tags(s: str) -> str:
    return _TAG_RE.sub("", s)


def _normalize_en(text: str, case_sensitive: bool = True) -> List[str]:
    """
    Word-level normalization (English, French, etc.)
    - strip tags
    - remove only puncts 
    - normalize whitespaces; optional case normalization
    - split by spaces
    """
    text = _strip_tags(text)
    text = "".join((" " if ch in _PUNCTS else ch) for ch in text)
    text = " ".join(text.split())
    if not case_sensitive:
        text = text.upper()
    return text.split() if text else []



def _normalize_zh(text: str) -> List[str]:
    """
    Char-level normalization (Chinese, Japanese, etc.)
    - strip tags
    - remove spaces and common puncts
    """
    text = _strip_tags(text)
    toks: List[str] = []
    for ch in text:
        if ch in _PUNCTS or ch in _SPACE_CHARS:
            continue
        cat = unicodedata.category(ch)
        if cat == 'Zs' or cat == 'Cn':
            continue
        toks.append(ch)
    return toks


def _levenshtein(ref: List[str], hyp: List[str]) -> Tuple[int, int, int, int]:
    """Return (S, D, I, N) for WER."""
    N = len(ref); M = len(hyp)
    dp = [[0] * (M + 1) for _ in range(N + 1)]
    for i in range(1, N + 1): dp[i][0] = i
    for j in range(1, M + 1): dp[0][j] = j
    for i in range(1, N + 1):
        ri = ref[i - 1]
        for j in range(1, M + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution / correct
            )
    i, j = N, M
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1; i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            I += 1; j -= 1
        else:
            if i > 0 and j > 0 and ref[i - 1] != hyp[j - 1]:
                S += 1
            i -= 1; j -= 1
    return S, D, I, N


def compute_wer_word(ref_text: str, hyp_text: str, case_sensitive: bool = True) -> float:
    """Word-level WER (English, French, etc.)."""
    ref = _normalize_en(ref_text, case_sensitive=case_sensitive)
    hyp = _normalize_en(hyp_text, case_sensitive=case_sensitive)
    S, D, I, N = _levenshtein(ref, hyp)
    return 0.0 if N == 0 else (S + D + I) / N


def compute_wer_char(ref_text: str, hyp_text: str) -> float:
    """Character-level WER (Chinese, Japanese, etc.)."""
    ref = _normalize_zh(ref_text)
    hyp = _normalize_zh(hyp_text)
    S, D, I, N = _levenshtein(ref, hyp)
    return 0.0 if N == 0 else (S + D + I) / N
