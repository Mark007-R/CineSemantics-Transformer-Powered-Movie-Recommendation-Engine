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


def embed_folder(model, processor, device, folder_path: str, batch_size: int = 32) -> Tuple[np.ndarray, List[Dict]]:
    if model is None or processor is None:
        logger.error("Model not loaded")
        raise ValueError("Model not loaded. Call load_clip_model() first.")
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        raise ValueError(f"Path is not a directory: {folder_path}")
    image_files = []
    for file_path in folder.rglob('*'):
        if file_path.is_file() and _validate_image_file(file_path):
            image_files.append(file_path.resolve())
    image_files = sorted(set(image_files))
    if len(image_files) == 0:
        logger.error(f"No valid images found in {folder_path}")
        raise ValueError(f"No valid images found in {folder_path}")
    logger.info(f"Found {len(image_files)} valid images")
    if batch_size < 1:
        logger.warning(f"Invalid batch_size {batch_size}, using 1")
        batch_size = 1
    elif batch_size > 128:
        logger.warning(f"Large batch_size {batch_size} may cause OOM, consider reducing")
    metadata = []
    for img_path in image_files:
        metadata.append({
            'image_path': str(img_path),
            'filename': img_path.name,
            'title': img_path.stem.replace('_', ' ').replace('-', ' ')
        })
    all_embeddings = []
    failed_count = 0
    successful_metadata = []
    logger.info("Extracting embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(metadata), batch_size), desc="Processing batches"):
            batch_data = metadata[i:i+batch_size]
            images = []
            batch_meta = []
            for item in batch_data:
                try:
                    img = _safe_open_image(Path(item['image_path']))
                    images.append(img)
                    batch_meta.append(item)
                except Exception as e:
                    failed_count += 1
                    continue
            if len(images) == 0:
                continue
            try:
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())
                successful_metadata.extend(batch_meta)
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                failed_count += len(images)
                continue
    if len(all_embeddings) == 0:
        logger.error("No embeddings were successfully extracted")
        raise RuntimeError("Failed to extract any embeddings")
    embeddings = np.vstack(all_embeddings)
    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} images")
    logger.info(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    return embeddings, successful_metadata