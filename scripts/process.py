from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path
from util.cv_utils import load_video, extractClips, proccess_Clip
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

# Paths
base_dir = Path(r"C:\Projects\Lip_Reading\GRID")
# Daniel's Path
# base_dir = Path(r"C:\Users\User\OneDrive\Documents\Projects\Lip-Reading\GRID")

# Command: python scripts/process.py s1 s2 ....
vocab = {
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

def process_video(video_file: Path, align_dir: Path, vocab: dict, save_dir: Path, parent_dir: Path):
    
    try:
        feat_seq, coords_seq, veloc_seq, acc_seq, labels = [], [], [], [], []
        align_file = align_dir / (video_file.stem + ".align")
        if not align_file.exists():
            print(f"Missing align file: {video_file.name}")
            return

        video = load_video(video_file)
        segments = extractClips(align_file)

        word_index = 0
        for start, end, word in segments:
            clip = video[start:end]
            try:
                feats, coords, veloc, acc = proccess_Clip(clip)
                
                feat_seq.append(np.array(feats, dtype=np.float32))
                coords_seq.append(np.array(coords, dtype=np.float32))
                veloc_seq.append(np.array(veloc, dtype=np.float32))
                acc_seq.append(np.array(acc, dtype=np.float32))
                labels.append(vocab[word])
                
                word_index += 1
            except Exception as e:
                print(f"Skipping {video_file.name} [{start}:{end}] due to error: {e}")
                continue

        save_path = save_dir / f"{video_file.stem}.pth"
        torch.save(
            {"x_feat": feat_seq, "x_coords": coords_seq, "x_veloc": veloc_seq, "x_acc": acc_seq,  "y_labels": labels},
            save_path
        )
        print(f"Processed: {video_file.name}")

    except Exception as e:
        print(f"Failed: {video_file.name} | Error: {e}")
    



 


# Run multiprocessed
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Optional but safe on Windows

    for i in range(1,len(sys.argv)):
        video_dir = base_dir / "raw/video_data" / sys.argv[i]
        align_dir = base_dir / "raw/alignment_data" / f"align_{sys.argv[i]}"
        
        # Proccesed data
        processed_dir = base_dir / "processed" / sys.argv[i]
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        parent_dir = Path(sys.argv[i])
        
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_video, video_path, align_dir, vocab, processed_dir, parent_dir)
                for video_path in video_dir.glob("*.mpg")
            ]

            # for future in futures:
            #     result = future.result() # return val for each process
            #     if result is None:
            #         continue
            #     for label, files in result.items():
            #         word_map[label].extend(files)
                
        print(f"Success: {sys.argv[i]}")

