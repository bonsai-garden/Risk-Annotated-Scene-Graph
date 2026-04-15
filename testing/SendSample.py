import argparse
import base64
import os
import sys

import cv2
import numpy as np
import requests

# Reuse loader via sys.path trick for a simple utility run
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.LoadDataset import DatasetLoader


def to_b64(img, ext=".png"):
    """Encode an image array to base64.

    Args:
        img: Image array.
        ext: Image extension for encoding.

    Returns:
        Base64-encoded string.
    """
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def main():
    """Send a single dataset sample to the Worker API.

    Args:
        None

    Returns:
        None
    """
    p = argparse.ArgumentParser(description="Send one dataset sample to Worker API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--path", default=None, help="Override dataset path")
    args = p.parse_args()

    loader = DatasetLoader(dataset_dir_override=args.path)
    ds = loader.load_dataset()
    if not ds:
        print("No samples found.")
        sys.exit(1)

    sample = ds[0]
    rgb = sample["rgb_img"]
    depth = sample["depth_img"]
    tx = sample["tx"].tolist() if hasattr(sample["tx"], "tolist") else list(sample["tx"])
    rotM = sample["rotM"].tolist()

    # Convert depth meters -> uint16 millimeters for transport
    depth_mm = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535).astype(np.uint16)

    payload = {
        "timestamp": sample["timestamp"],
        "rgb": to_b64(img=rgb, ext=".jpg"),
        "depth": to_b64(img=depth_mm, ext=".png"),
        "tx": tx,
        "rotM": rotM,
    }

    url = f"http://{args.host}:{args.port}"
    print("POST /frame ...")
    r = requests.post(f"{url}/frame", json=payload, timeout=30)
    print("Response:", r.status_code, r.text)

    print("GET /scenegraph ...")
    r = requests.get(f"{url}/scenegraph", timeout=30)
    print("Scenegraph nodes:", len(r.json().get("nodes", [])))


if __name__ == "__main__":
    main()
