"""
Load training images into Visual RAG database (ChromaDB).

Usage:
    python RAG_DataLoader.py --path /path/to/training/data

Structure:
    /path/to/training/data/
        /firehydrant/
            image1.jpg
            image2.jpg
            ...
        /tablet/
            image1.jpg
            image2.jpg
            ...
        /keyboard/
            ...

Recommendation:
    5-15 images per class for good detection confidence.
    More images = better accuracy, but diminishing returns after ~20 per class.
    
Note:
    ChromaDB is persistent (stored in visual_memory/ directory).
    You only need to run this ONCE. The database will be reused for all inference.
"""

import sys
import os
import argparse
from pathlib import Path
from PIL import Image
from scripts.VisualRAP import VisualRAP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_training_data(data_dir: str):
    """Load labeled training images into the Visual RAG database.

    Args:
        data_dir: Path to the training data root folder.

    Returns:
        None
    """
    visual_rag = VisualRAP()
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: {data_dir} does not exist")
        return
    
    for class_folder in sorted(data_path.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        image_count = 0
        
        for image_file in sorted(class_folder.glob("*")):
            if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            
            try:
                image = Image.open(image_file).convert("RGB")
                visual_rag.add_image(image=image, label=class_name)
                image_count += 1
            except Exception as e:
                print(f"Warning: Failed to load {image_file}: {e}")
        
        if image_count > 0:
            print(f"[RAG] Loaded {image_count} images for class: {class_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load training data into Visual RAG")
    parser.add_argument("--path", required=True, help="Path to training data folder")
    args = parser.parse_args()
    print("[RAG] Loading training data...")
    load_training_data(args.path)
    print("[RAG] Training complete.")

