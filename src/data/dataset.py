"""
Dataset class for handling code review data
Basically loads the processed data and returns it in a format the model can use

************************might need to add data augmentation later?**************************
Also should probably add some validation checks...
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .processors import CodeProcessor, DiffImageProcessor, ContextProcessor
from ..config import config


class CodeReviewDataset(Dataset):
    """
    Custom dataset for my code review summarizer
    Combines all the processed data into something PyTorch can understand
    
    *************check out how huggingface datasets work, might be able to use some of their features********************
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[callable] = None  # for image augmentation maybe
    ):
        """
        Sets up the dataset
        
        Args:
            data_dir: where the processed data lives
            split: either "train" or "val" (might add "test" later)
            transform: stuff to do with the images (like resize etc)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        #create our processor objects
        self.code_proc = CodeProcessor()
        self.img_proc = DiffImageProcessor()
        self.ctx_proc = ContextProcessor()
        
        #load all our data
        self.samples = self._load_data()
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_data(self) -> List[Dict]:
        """
        Reads the json data file and checks if it's valid
        
        Returns:
            list of data samples (each one is a dict with diff, context etc)
        """
        samples = []
        data_file = self.data_dir / f"{self.split}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Can't find data file: {data_file}")
        
        #load json data
        print(f"Loading data from {data_file}")
        with open(data_file, "r") as f:
            data = json.load(f)
        
        for item in data:
            # need these fields for the model to work
            needed_fields = ["diff", "context", "summary"]
            if not all(field in item for field in needed_fields):
                print(f"Skipping sample - missing fields: {item.keys()}")
                continue
            
            samples.append(item)
        
        return samples
    
    def __len__(self) -> int:
        """How many samples we have"""
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Gets one sample and processes it for the model
        
        Args:
            idx: which sample to get
            
        Returns:
            dict with all the processed stuff:
            - diff_image: the visualization
            - diff_text: the actual diff
            - context: PR description etc
            - summary: what we're trying to predict
        """

        sample = self.samples[idx]
        
        diff = self.code_proc.process_diff(sample["diff"])
        
        #visualization
        img = self.img_proc.diff_to_image(diff)
        if self.transform:
            img = self.transform(img)
        
        ctx = self.ctx_proc.process_context(
            title=sample.get("title", ""),  #use empty string if missing
            desc=sample.get("description", ""),
            comments=sample.get("comments", [])
        )
        
        #*********need to change image format for pytorch
        return {
            "diff_image": torch.from_numpy(img).permute(2, 0, 1),  # HWC -> CHW
            "diff_text": diff,
            "context": ctx,
            "summary": sample["summary"]
        }
