import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.absolute()


@dataclass
class DataConfig:
    """Data processing configuration."""
    raw_data_dir: Path = ROOT_DIR / "data" / "raw"
    processed_data_dir: Path = ROOT_DIR / "data" / "processed"
    max_code_length: int = 512
    max_context_length: int = 256
    image_size: tuple[int, int] = (224, 224)
    train_test_split: float = 0.2


@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    num_workers: int = os.cpu_count() or 4
    batch_size: int = 32
    prefetch_factor: int = 2


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = DataConfig()
    processing: ProcessingConfig = ProcessingConfig()
    seed: int = 42
    device: str = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
    debug: bool = os.environ.get("DEBUG", "0") == "1"


# Global config instance
config = Config()
