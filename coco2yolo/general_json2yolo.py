# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import json
from collections import defaultdict

import cv2
import pandas as pd
from PIL import Image

from coco2yolo.utils import *


def convert_coco_json(json_dir="../coco/annotations/", labels_dir="new_dir/", cls91to80=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs(labels_dir)  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"{:g}".format(x["id"]): x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).

    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def delete_dsstore(path="../datasets"):
    """Deletes Apple .DS_Store files recursively from a specified directory."""
    from pathlib import Path

    files = list(Path(path).rglob(".DS_store"))
    print(files)
    for f in files:
        f.unlink()