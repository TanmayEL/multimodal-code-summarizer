"""
This module handles all the data processing stuff for my project.
Main components:
- CodeProcessor: handles code and diff processing
- DiffImageProcessor: converts diffs to visualizations
- ContextProcessor: processes PR descriptions and stuff

***************shld add more preprocessing steps later******************************************************************
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..config import config


class CodeProcessor:
    """
    Handles code processing - removes git stuff from diffs and does basic tokenization
    """
    
    def __init__(self, max_len = config.data.max_code_length):
        self.max_len = max_len
    
    def process_diff(self, diff_txt) -> str:
        #cleans up the diff text by removing git metadata and stuff

        lines = diff_txt.split("\n")
        clean_lines = []
        
        for line in lines:
            #skip git metadata
            if line.startswith("diff --git") or line.startswith("index"):
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            clean_lines.append(line)
        
        return "\n".join(clean_lines)
    
    def tokenize_code(self, code) -> List[str]:
        """
        basic tokenization - just splits on whitespace for now might need something better later but this works ok
        """
        tokens = code.split()
        return tokens[:self.max_len]  # truncate if too long


class DiffImageProcessor:
    """
    Converts diffs into images kinda like Github's diff view but as a matrix
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = config.data.image_size,
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None
    ):
        self.img_size = img_size
        
        #colors for different parts of the diff
        self.colors = colors or {
            "+": (0, 255, 0),  # green = added stuff
            "-": (255, 0, 0),  # red = deleted stuff
            "@": (0, 0, 255),  # blue = where changes happen
            " ": (255, 255, 255)  # white = unchanged code
        }
    
    def diff_to_image(self, diff_txt: str) -> np.ndarray:
        lines = diff_txt.split("\n")
        h, w = self.img_size
        
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        num_lines = min(len(lines), h)  #dont go over image height
        if num_lines == 0:
            return img
            
        line_h = h // num_lines
        
        for i, line in enumerate(lines[:h]):  #only process lines that fit
            if not line:  #skip empty lines
                continue
                
            #pick color based on first char (+ - @ or space)
            first_char = line[0] if line else " "
            color = self.colors.get(first_char, self.colors[" "])
            
            y_pos = i * line_h
            
            cv2.rectangle(
                img,
                (0, y_pos),
                (w, y_pos + line_h),
                color,
                thickness=-1  # -1 means fill
            )
            
            visible_text = line[:w//10]  #show what fits
            cv2.putText(
                img,
                visible_text,
                (5, y_pos + line_h - 5),  #position near bottom of barr
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  #works
                (0, 0, 0),
                1
            )
        
        return img


class ContextProcessor:
    """
    Handles all the PR text stuff - title, description, and comments
    Basically combines everything into one text that the model can understand
    
    Note: might need to handle markdown formatting better?
    Also the max length thing is kinda hacky but works for now
    """
    
    def __init__(self, max_len: int = config.data.max_context_length):
        #max number of words to keep (to avoid super long inputs)
        self.max_len = max_len
    
    def process_context(self, title: str, description: str, comments: List[str]) -> str:
        context = f"Title: {title}\nDescription: {description}"
        
        if comments:
            context += "\nComments:\n" + "\n".join(comments)
        
        words = context.split()
        return " ".join(words[:self.max_len])  # truncate if needed
