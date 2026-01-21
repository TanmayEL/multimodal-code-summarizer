import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import config
from src.data.processors import CodeProcessor, DiffImageProcessor, ContextProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_raw_data(raw_data_path: Path, processed_data_path: Path) -> Tuple[List[Dict], List[Dict]]:
    #Process raw data and split into train and validation sets.
    #start processors
    code_processor = CodeProcessor()
    image_processor = DiffImageProcessor()
    context_processor = ContextProcessor()
    
    data = []
    raw_files = list(raw_data_path.glob("*.json"))
    
    for file_path in tqdm(raw_files, desc="Processing files"):
        with open(file_path, "r") as f:
            items = json.load(f)
            
        for item in items:
            try:
                processed_diff = code_processor.process_diff(item["diff"])
                
                #generate diff imagess
                diff_image = image_processor.diff_to_image(processed_diff)
                
                context = context_processor.process_context(
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    comments=item.get("comments", [])
                )
                
                # Create proccessed samples
                processed_item = {
                    "diff": processed_diff,
                    "context": context,
                    "summary": item["summary"]
                }
                
                data.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                continue
    
    #split data
    train_data, val_data = train_test_split(
        data,
        test_size=config.data.train_test_split,
        random_state=config.seed
    )
    
    return train_data, val_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(config.data.raw_data_dir),
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(config.data.processed_data_dir),
        help="Directory to save processed data"
    )
    
    args = parser.parse_args()
    
    raw_path = Path(args.raw_dir)
    processed_path = Path(args.processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    logger.info("Processing data...")
    train_data, val_data = process_raw_data(raw_path, processed_path)

    logger.info("Saving processed data...")
    with open(processed_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(processed_path / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    logger.info(f"Processed {len(train_data)} train and {len(val_data)} validation samples.")


if __name__ == "__main__":
    main()
