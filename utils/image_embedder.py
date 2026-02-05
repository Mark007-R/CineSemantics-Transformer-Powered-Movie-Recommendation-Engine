import logging
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_clip_model(model_name='openai/clip-vit-base-patch32', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading CLIP model: {model_name} on {device}")
    try:
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()
        logger.info("Model loaded successfully")
        return model, processor, device
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def _validate_image_file(file_path: Path) -> bool:
    """Validate if file is a supported image format."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    return file_path.suffix.lower() in valid_extensions


def _safe_open_image(image_path: Path) -> Image.Image:
    try:
        img = Image.open(image_path).convert('RGB')
        if img.size[0] < 1 or img.size[1] < 1:
            raise ValueError(f"Invalid image dimensions: {img.size}")
        return img
    except Exception as e:
        logger.warning(f"Error loading {image_path.name}: {e}")
        raise
