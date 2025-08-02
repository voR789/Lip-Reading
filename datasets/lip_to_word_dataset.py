import numpy as np
import os
import glob
import torch
from torch.utils.data import Dataset
from pathlib import Path

class LipReadingWordDataset(Dataset):
    # Implement abstract funcs
    
    # constructor
    # pass in path object of proccessed folder
    def __init__(self, processed_dir):
        self.index_map = []
        self.cache = {} # data optimization as we save loading times for redudant multiple get item calls for data, etc: epochs
        processed_dir = Path(processed_dir)
        for folder in processed_dir.iterdir():
            folder = Path(folder)
            for data_file in folder.glob("*.pth"):            
                data = torch.load(data_file, map_location= 'cpu', weights_only=False)
                num_words = len(data["y_labels"])
                for i in range(num_words):
                    self.index_map.append((data_file, i)) 
                    # load in the indicies and path of each word token, and load actual data in real time for optimization

    def __len__(self):
        # Return total amount of word tokens
        return len(self.index_map)

    def __getitem__(self, idx):
        # Return part of sequence for each word using index map (index being which word in the file)
        data_path, index = self.index_map[idx]
        if data_path not in self.cache:
            self.cache[data_path] = torch.load(data_path, map_location= 'cpu', weights_only= False)
        # load from cache
        data = self.cache[data_path]
        
        # Load data into tensors at runtime
        x_feat, x_coords, x_veloc, y_labels = data["x_feat"][index], data["x_coords"][index], data["x_veloc"][index], data["y_labels"][index]
        return torch.tensor(x_feat, dtype= torch.float32), torch.tensor(x_coords, dtype= torch.float32), torch.tensor(x_veloc, dtype= torch.float32), torch.tensor(y_labels, dtype= torch.long)
        
