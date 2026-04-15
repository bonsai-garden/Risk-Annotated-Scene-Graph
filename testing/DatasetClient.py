"""
DatasetClient - Sends dataset frames to the Worker API in a continuous stream.

DATASET STRUCTURE REQUIREMENTS:
================================
Your dataset directory should follow this structure:

    dataset_dir/
    ├── rgb/                      # RGB images (PNG format)
    │   ├── 1305031455.png       # Timestamp as filename (without extension)
    │   ├── 1305031458.png
    │   └── ...
    ├── depth/                    # Depth images (PNG format)
    │   ├── 1305031455.png       # Must match RGB filename (same timestamp)
    │   ├── 1305031458.png
    │   └── ...
    └── groundtruth.txt          # Pose data (see format below)


FILE NAMING CONVENTIONS:
=======================
1. RGB Images:
   - Location: dataset_dir/rgb/
   - Format: PNG files
   - Naming: Use timestamp as filename (no spaces, no special chars)
   - Example: 1305031455.png (represents timestamp 1305031455)

2. Depth Images:
   - Location: dataset_dir/depth/
   - Format: PNG files (16-bit, values in mm)
   - Naming: MUST match RGB filename exactly (same timestamp)
   - Example: 1305031455.png
   - Note: Depth values are typically in millimeters (0-65535 range for uint16)

3. Timestamps:
   - RGB and Depth files MUST have matching timestamps
   - Timestamps are used as keys to pair images with pose data
   - Files are loaded in sorted order


GROUNDTRUTH.TXT FILE FORMAT:
=============================
The groundtruth.txt file contains camera pose (position + orientation) for each frame.
It must be located in the dataset root directory.

Format (space-separated):
    timestamp tx ty tz qx qy qz qw

Where:
- timestamp: Frame identifier (must match RGB/Depth filenames, e.g., 1305031455)
- tx, ty, tz: Translation vector (camera position in meters)
- qx, qy, qz, qw: Quaternion representing rotation (normalized)
  * Format: [x, y, z, w]
  * Must be normalized (length = 1.0)
  * Converted internally to 3x3 rotation matrix

Example groundtruth.txt:
    # Time tx ty tz qx qy qz qw
    # Dataset timestamp 1305031453.873
    1305031455 0.0 0.0 0.0 0.0 0.0 0.0 1.0
    1305031458 0.1 0.2 -0.5 0.1 0.2 0.3 0.9
    1305031463 0.2 0.4 -1.0 0.2 0.3 0.4 0.8
    1305031468 0.3 0.6 -1.5 0.3 0.4 0.5 0.7

Notes:
- Lines starting with '#' are ignored as comments
- Each timestamp in groundtruth.txt must have corresponding RGB and Depth files
- Files without matching groundtruth entries will be skipped


USAGE EXAMPLES:
===============
Default dataset (from config/parameters.py):
    python testing/DatasetClient.py

Custom dataset path:
    python testing/DatasetClient.py --path "/path/to/custom/dataset"

With custom timing and target server:
    python testing/DatasetClient.py --host 192.168.1.100 --port 8000 --interval 2

Continuous looping (restart after all frames):
    python testing/DatasetClient.py --repeat

Full example with all options:
    python testing/DatasetClient.py \\
        --path "data/dataset_test" \\
        --host "127.0.0.1" \\
        --port 8000 \\
        --interval 1.5 \\
        --repeat
"""

import argparse
import base64
import os
import sys
import time

import cv2
import numpy as np
import requests

from scripts.LoadDataset import DatasetLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TIMEOUT_SEND = 30
TIMEOUT_STATUS = 5
TIMEOUT_HEALTH = 5

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
    """Send dataset frames to the Worker API in a loop.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--path", default=None)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()

    print(f"Loading dataset from {args.path or 'default location'}...")
    loader = DatasetLoader(dataset_dir_override=args.path)
    ds = loader.load_dataset()
    if not ds:
        print("Error: No samples found in dataset.")
        sys.exit(1)

    print(f"Loaded {len(ds)} samples.")
    print(f"Connecting to Worker API at http://{args.host}:{args.port}")

    try:
        r = requests.get(f"http://{args.host}:{args.port}/health", timeout=TIMEOUT_HEALTH)
        if r.status_code == 200:
            print("Connected to Worker API\n")
        else:
            print(f"Worker API returned status {r.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to connect to Worker API: {e}")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    frame_count = 0

    try:
        while True:
            for sample_idx, sample in enumerate(ds):
                frame_count += 1
                
                rgb = sample["rgb_img"]
                depth = sample["depth_img"]
                tx = (
                    sample["tx"].tolist()
                    if hasattr(sample["tx"], "tolist")
                    else list(sample["tx"])
                )
                rotM = sample["rotM"].tolist()

                depth_mm = np.clip(
                    np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535
                ).astype(np.uint16)

                payload = {
                    "timestamp": sample["timestamp"],
                    "rgb": to_b64(img=rgb, ext=".jpg"),
                    "depth": to_b64(img=depth_mm, ext=".png"),
                    "tx": tx,
                    "rotM": rotM,
                }

                try:
                    print(
                        f"[{frame_count}] Sending frame {sample_idx + 1}/{len(ds)} "
                        f"(timestamp: {sample['timestamp']})..."
                    )
                    r = requests.post(
                        f"{url}/frame", json=payload, timeout=TIMEOUT_SEND
                    )
                    
                    if r.status_code == 200:
                        result = r.json()
                        frame_id = result.get("frame_id")
                        queue_size = result.get("queue_size", 0)
                        print(
                            f"  Queued (frame_id={frame_id}, queue_size={queue_size})"
                        )
                    else:
                        print(f"  Error: HTTP {r.status_code}")
                        print(f"    Response: {r.text[:200]}")

                except requests.exceptions.Timeout:
                    print(f"  Timeout sending frame")
                except requests.exceptions.ConnectionError:
                    print(f"  Connection error - is Worker API running?")
                    sys.exit(1)
                except Exception as e:
                    print(f"  Error: {e}")

                if sample_idx < len(ds) - 1 or args.repeat:
                    time.sleep(args.interval)

            if not args.repeat:
                print("\nAll frames sent. Exiting.")
                break
            else:
                print(f"\nDataset loop complete. Restarting in {args.interval}s...")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        
        try:
            r = requests.get(f"{url}/queue/status", timeout=TIMEOUT_STATUS)
            if r.status_code == 200:
                stats = r.json()
                print("\nFinal Queue Statistics:")
                print(f"  Total frames sent: {frame_count}")
                print(f"  Queue size: {stats['queue_size']}")
                print(f"  Completed: {stats['completed']}")
                print(f"  Processing: {stats['processing']}")
                print(f"  Failed: {stats['failed']}")
        except Exception as e:
            print(f"Could not retrieve final stats: {e}")


if __name__ == "__main__":
    main()
