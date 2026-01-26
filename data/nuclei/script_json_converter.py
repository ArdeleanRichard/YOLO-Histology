"""
Simple YOLO -> COCO GT converter (no argparse, only top variables to change).

Assumptions:
 - One .txt label file per image, same stem.
 - YOLO label lines: class x_center y_center width height (normalized 0..1).
 - Images are in IMAGES_DIR (jpg/png...). If USE_FILENAME_AS_ID=True, image filenames
   must be integer stems (e.g. 12345.jpg) and those integers will be used as COCO image_id.
"""

import json
from pathlib import Path
from PIL import Image


IMAGES_DIR = Path("./images/test/")                     # path to images
LABELS_DIR = Path("./labels/test/")                     # path to YOLO .txt files
OUTPUT_JSON = Path("./ground_truth_labels_test.json")  # output COCO JSON path

CLASS_NAMES_FILE = None      # Path to class names file (one name per line), or None
CATEGORY_ID_OFFSET = 1       # 0 if your classes are 0-based and you want category_id==class,
                            # 1 if you want COCO-style 1-based category ids
USE_FILENAME_AS_ID = False   # If True, uses integer image filename stem as image_id
STARTING_IMAGE_ID = 1
STARTING_ANN_ID = 1
SKIP_MISSING_IMAGES = True
# ---------------------------------------------------



def find_image_for_label(label_stem, images_dir):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    for ext in exts:
        cand = images_dir / (label_stem + ext)
        if cand.exists():
            return cand
    # fallback: try a file with same stem ignoring small differences
    for img in images_dir.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in exts:
            continue
        if img.stem == label_stem or img.stem.startswith(label_stem) or label_stem.startswith(img.stem):
            return img
    return None

def convert_yolo_to_coco_bbox(xc, yc, w, h, img_w, img_h):
    bw = w * img_w
    bh = h * img_h
    x0 = xc * img_w - bw / 2.0
    y0 = yc * img_h - bh / 2.0
    if x0 < 0: x0 = 0.0
    if y0 < 0: y0 = 0.0
    return [round(x0, 3), round(y0, 3), round(bw, 3), round(bh, 3)]

def load_class_names(fn):
    if not fn:
        return None
    p = Path(fn)
    if not p.exists():
        print(f"Class names file not found: {fn}")
        return None
    with open(p, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

def main():
    images_dir = IMAGES_DIR
    labels_dir = LABELS_DIR
    out_json = OUTPUT_JSON

    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")
    if not images_dir.exists():
        raise SystemExit(f"Images directory not found: {images_dir}")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(CLASS_NAMES_FILE)
    categories = []
    if class_names:
        for idx, n in enumerate(class_names):
            categories.append({"id": idx + CATEGORY_ID_OFFSET, "name": n})

    images = []
    annotations = []
    ann_id = STARTING_ANN_ID
    next_image_id = STARTING_IMAGE_ID
    seen_category_ids = set()

    label_files = sorted(labels_dir.glob("*.txt"))
    if len(label_files) == 0:
        print("No .txt label files found in", labels_dir)
        return

    for lbl in label_files:
        stem = lbl.stem
        img_path = find_image_for_label(stem, images_dir)
        if img_path is None:
            msg = f"No image found for label {lbl.name}"
            if SKIP_MISSING_IMAGES:
                print("Warning:", msg, "-> skipping")
                continue
            else:
                raise SystemExit(msg)

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        if USE_FILENAME_AS_ID:
            try:
                image_id = int(img_path.stem)
            except Exception:
                print(f"Filename stem not integer for {img_path.name}; using incremental id.")
                image_id = next_image_id
                next_image_id += 1
        else:
            image_id = next_image_id
            next_image_id += 1

        images.append({
            "id": img_path.stem, # image_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h
        })

        # read label lines
        with open(lbl, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                print(f"Skipping malformed line in {lbl.name}: {ln}")
                continue
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            bbox = convert_yolo_to_coco_bbox(xc, yc, bw, bh, img_w, img_h)
            area = round(bbox[2] * bbox[3], 3)
            category_id = cls + CATEGORY_ID_OFFSET
            seen_category_ids.add(category_id)

            annotations.append({
                "id": ann_id,
                "image_id": img_path.stem, # image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })
            ann_id += 1

    # if categories not provided, infer
    if not categories:
        cats = sorted(list(seen_category_ids))
        categories = [{"id": cid, "name": f"class_{cid - CATEGORY_ID_OFFSET}"} for cid in cats]

    coco = {
        "info": {"description": "Converted from YOLO labels", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print(f"WROTE: {out_json}")
    print(f"Images: {len(images)}, Annotations: {len(annotations)}, Categories: {len(categories)}")

if __name__ == "__main__":
    main()
