import torch
import numpy as np
from util.cv_utils import load_video, extractClips, proccess_Clip
from pathlib import Path
import sys


import warnings
warnings.filterwarnings("ignore")

# TO USE RUN IN TERMINAL, along with sample folder name:
# Ex) python scripts/process.py s1
# File Paths
base_path, video_subdir, align_subdir, processed_subdir = r"C:\Projects\Lip_Reading\GRID", r"raw/video_data",  r"raw/alignment_data", r"processed"
base_dir = Path(base_path)
video_dir = base_dir / video_subdir / sys.argv[1]
align_dir = base_dir / align_subdir / f"align_{sys.argv[1]}"

# Dataset Vocab
vocab = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "sp": 3,
    "bin": 4,
    "lay": 5,
    "place": 6,
    "set": 7,
    "blue": 8,
    "green": 9,
    "red": 10,
    "white": 11,
    "at": 12,
    "by": 13,
    "in": 14,
    "with": 15,
    "zero": 16,
    "one": 17,
    "two": 18,
    "three": 19,
    "four": 20,
    "five": 21,
    "six": 22,
    "seven": 23,
    "eight": 24,
    "nine": 25,
    "again": 26,
    "now": 27,
    "please": 28,
    "soon": 29,
    "a": 30,
    "b": 31,
    "c": 32,
    "d": 33,
    "e": 34,
    "f": 35,
    "g": 36,
    "h": 37,
    "i": 38,
    "j": 39,
    "k": 40,
    "l": 41,
    "m": 42,
    "n": 43,
    "o": 44,
    "p": 45,
    "q": 46,
    "r": 47,
    "s": 48,
    "t": 49,
    "u": 50,
    "v": 51,
    "x": 52,
    "y": 53,
    "z": 54
}
# Video by video approach: 
# Memory Efficiency: You can load data lazily—only what you need per batch—so you’re not forced to load all data at once.
try:
    for video_file in video_dir.glob("*.mpg"):
        feat_seq, coords_seq, veloc_seq, labels = [], [], [], []    
        align_file = align_dir / (video_file.stem + ".align")
        if(not align_file.exists()):
            raise RuntimeError(f"Align path does not exist for: {video_file.name}")
        video = load_video(video_file)
        segments = extractClips(align_file)
        for start, end, word in segments:
            clip = video[start:end]
            try:
                feats, coords, veloc = proccess_Clip(clip)
            except Exception as e:
                print(f"Skipping segment {video_file.stem} [{start}:{end}] due to error: {e}")       
                continue 
            feat_seq.append(np.array(feats, dtype=np.float32))
            coords_seq.append(np.array(coords, dtype=np.float32))
            veloc_seq.append(np.array(veloc, dtype=np.float32))
            labels.append(vocab[word])
        
        # Save to data files
        processed_dir = base_dir / processed_subdir / f"{sys.argv[1]}" 
        processed_dir.mkdir(parents=True, exist_ok=True) # create dir
        processed_file = processed_dir / f"{video_file.stem}_data.pth"
        
        # data usage (.pth)
        torch.save(
            {"x_feat": feat_seq, "x_coords": coords_seq, "x_veloc": veloc_seq, "y_labels": labels},
            processed_file
            )
except Exception as e:
    print(f"Failed to process {video_file.name}. {e}")
print("Success!")