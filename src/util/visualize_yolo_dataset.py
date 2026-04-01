import argparse
import random
from pathlib import Path

import cv2
import yaml

try:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
except ImportError:
    get_cfg = None
    build_dataloader = None
    build_yolo_dataset = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_label_line(line: str):
    raw = line.strip()
    if not raw:
        return None, "empty_line"

    parts = raw.split()
    if len(parts) < 5:
        return None, f"too_few_fields:{len(parts)}"

    try:
        class_id = int(float(parts[0]))
        values = [float(x) for x in parts[1:]]
    except ValueError:
        return None, "non_numeric"

    if len(values) < 4:
        return None, "missing_bbox"

    bbox = values[:4]
    remain = values[4:]
    if len(remain) % 2 != 0:
        return None, f"odd_keypoint_fields:{len(remain)}"

    keypoints = [(remain[i], remain[i + 1]) for i in range(0, len(remain), 2)]
    return {"class_id": class_id, "bbox": bbox, "keypoints": keypoints}, None


def _bbox_xyxy_from_yolo(bbox, img_w: int, img_h: int):
    cx, cy, bw, bh = bbox
    x1 = (cx - bw / 2.0) * img_w
    y1 = (cy - bh / 2.0) * img_h
    x2 = (cx + bw / 2.0) * img_w
    y2 = (cy + bh / 2.0) * img_h
    return x1, y1, x2, y2


def _is_bbox_norm_invalid(bbox):
    cx, cy, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return True
    if cx < 0 or cx > 1 or cy < 0 or cy > 1:
        return True
    if bw > 1 or bh > 1:
        return True
    return False


def _is_kpt_norm_invalid(kpt):
    x, y = kpt
    return x < 0 or x > 1 or y < 0 or y > 1


def _draw_instance(img, parsed, line_idx: int, invalid: bool):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _bbox_xyxy_from_yolo(parsed["bbox"], w, h)

    color = (0, 0, 255) if invalid else (0, 220, 0)
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(img, p1, p2, color, 2)

    tag = f"cls={parsed['class_id']} line={line_idx}"
    if invalid:
        tag += " INVALID"

    text_org = (max(0, p1[0]), max(18, p1[1] - 6))
    cv2.putText(img, tag, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for kx, ky in parsed["keypoints"]:
        px = int(round(kx * w))
        py = int(round(ky * h))
        cv2.circle(img, (px, py), 3, (255, 140, 0), -1)


def _draw_instance_from_xywhn(img, class_id: int, bbox, line_idx: int):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _bbox_xyxy_from_yolo(bbox, w, h)
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    color = (255, 80, 80)
    cv2.rectangle(img, p1, p2, color, 2)
    tag = f"cls={class_id} dl_idx={line_idx}"
    text_org = (max(0, p1[0]), max(18, p1[1] - 6))
    cv2.putText(img, tag, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _tensor_img_to_bgr(img_chw):
    # Ultralytics dataloader returns uint8 CHW RGB images.
    img_hwc = img_chw.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)


def _render_manual_image(image_path: Path, label_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, True, 0, 0

    invalid_in_image = False
    instances_total = 0
    instances_invalid = 0

    if not label_path.exists():
        cv2.putText(
            img,
            "NO LABEL FILE",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img, True, 0, 0

    with label_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        parsed, err = _parse_label_line(line)
        if err is not None:
            invalid_in_image = True
            instances_invalid += 1
            cv2.putText(
                img,
                f"PARSE ERROR line {i}: {err}",
                (16, 24 + 20 * min(i, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            continue

        instances_total += 1
        invalid = _is_bbox_norm_invalid(parsed["bbox"]) or any(
            _is_kpt_norm_invalid(kpt) for kpt in parsed["keypoints"]
        )
        if invalid:
            instances_invalid += 1
            invalid_in_image = True
        _draw_instance(img, parsed, i, invalid)

    return img, invalid_in_image, instances_total, instances_invalid


def _render_dataloader_map(
    dataset_root: Path,
    data_yaml: Path,
    split: str,
    batch_size: int,
    workers: int,
    imgsz: int,
    loader_mode: str,
    max_images: int | None,
):
    if get_cfg is None or build_yolo_dataset is None or build_dataloader is None:
        raise ImportError(
            "ultralytics is not installed in current environment. Please run with your YOLO environment."
        )

    with data_yaml.open("r", encoding="utf-8") as f:
        data_dict = yaml.safe_load(f)

    cfg = get_cfg()
    cfg.task = "pose"
    cfg.imgsz = imgsz

    img_dir = dataset_root / "images" / split
    dataset = build_yolo_dataset(
        cfg,
        str(img_dir),
        batch=batch_size,
        data=data_dict,
        mode=loader_mode,
        rect=False,
        stride=32,
    )
    dataloader = build_dataloader(dataset, batch=batch_size, workers=workers, shuffle=False)

    rendered = {}
    count = 0
    for batch in dataloader:
        imgs = batch["img"]
        bboxes = batch["bboxes"].cpu().numpy()
        classes = batch["cls"].cpu().numpy().reshape(-1)
        batch_idx = batch["batch_idx"].cpu().numpy()
        im_files = [str(Path(x).resolve()) for x in batch["im_file"]]

        for i, im_file in enumerate(im_files):
            out = _tensor_img_to_bgr(imgs[i])
            inds = (batch_idx == i).nonzero()[0]
            for j, label_i in enumerate(inds, start=1):
                cls_id = int(round(float(classes[label_i])))
                _draw_instance_from_xywhn(out, cls_id, bboxes[label_i], j)

            cv2.putText(out, "DATALOADER VIEW", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
            rendered[im_file] = out
            count += 1
            if max_images is not None and count >= max_images:
                return rendered

    return rendered


def _stack_compare(left_img, right_img):
    lh, lw = left_img.shape[:2]
    rh, rw = right_img.shape[:2]
    if rh != lh or rw != lw:
        right_img = cv2.resize(right_img, (lw, lh), interpolation=cv2.INTER_LINEAR)
    return cv2.hconcat([left_img, right_img])


def visualize_yolo_dataset(
    dataset_root: Path,
    data_yaml: Path,
    split: str,
    output_dir: Path,
    max_images: int | None,
    shuffle: bool,
    seed: int,
    mode: str,
    batch_size: int,
    workers: int,
    imgsz: int,
    loader_mode: str,
):
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    image_paths.sort()

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(image_paths)

    if max_images is not None and max_images > 0:
        image_paths = image_paths[:max_images]

    stats = {
        "images_checked": 0,
        "images_without_label": 0,
        "images_with_invalid_label": 0,
        "instances_total": 0,
        "instances_invalid": 0,
        "parse_errors": 0,
    }

    dataloader_map = {}
    if mode in {"dataloader", "compare"}:
        dataloader_map = _render_dataloader_map(
            dataset_root=dataset_root,
            data_yaml=data_yaml,
            split=split,
            batch_size=batch_size,
            workers=workers,
            imgsz=imgsz,
            loader_mode=loader_mode,
            max_images=max_images,
        )

    for image_path in image_paths:
        stats["images_checked"] += 1
        label_path = labels_dir / f"{image_path.stem}.txt"

        manual_img, invalid_in_image, inst_total, inst_invalid = _render_manual_image(image_path, label_path)
        if manual_img is None:
            stats["images_with_invalid_label"] += 1
            continue

        if not label_path.exists():
            stats["images_without_label"] += 1
        stats["instances_total"] += inst_total
        stats["instances_invalid"] += inst_invalid

        if invalid_in_image:
            stats["images_with_invalid_label"] += 1

        if mode == "manual":
            out_img = manual_img
        elif mode == "dataloader":
            dl_img = dataloader_map.get(str(image_path.resolve()))
            if dl_img is None:
                out_img = manual_img.copy()
                cv2.putText(
                    out_img,
                    "NO DATALOADER SAMPLE",
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                out_img = dl_img
        else:  # compare
            dl_img = dataloader_map.get(str(image_path.resolve()))
            if dl_img is None:
                out_img = manual_img.copy()
                cv2.putText(
                    out_img,
                    "NO DATALOADER SAMPLE",
                    (16, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    manual_img,
                    "MANUAL LABEL VIEW",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                out_img = _stack_compare(manual_img, dl_img)

        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), out_img)

    print("YOLO dataset visualization finished")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"saved_dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO labels on images and report malformed/out-of-range annotations."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Dataset root containing images/<split> and labels/<split>.",
    )
    parser.add_argument(
        "--data_yaml",
        type=Path,
        required=True,
        help="Dataset YAML (e.g. dataset/ballmany.yaml) used by Ultralytics dataloader.",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train/val/test.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("show/yolo_dataset_vis"),
        help="Directory to save rendered images.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=200,
        help="Max number of images to visualize. Use <=0 to process all.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle image order before sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle.")
    parser.add_argument(
        "--mode",
        choices=["manual", "dataloader", "compare"],
        default="compare",
        help="manual: parse txt directly; dataloader: render YOLO dataloader output; compare: side-by-side.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for YOLO dataloader mode.")
    parser.add_argument("--workers", type=int, default=0, help="Workers for YOLO dataloader.")
    parser.add_argument("--imgsz", type=int, default=640, help="Dataloader image size (usually training size).")
    parser.add_argument(
        "--loader_mode",
        choices=["train", "val"],
        default="val",
        help="Ultralytics dataset mode. Use val to avoid train-time augment when debugging labels.",
    )
    args = parser.parse_args()

    max_images = args.max_images if args.max_images and args.max_images > 0 else None
    visualize_yolo_dataset(
        dataset_root=args.dataset_root,
        data_yaml=args.data_yaml,
        split=args.split,
        output_dir=args.output_dir,
        max_images=max_images,
        shuffle=args.shuffle,
        seed=args.seed,
        mode=args.mode,
        batch_size=args.batch_size,
        workers=args.workers,
        imgsz=args.imgsz,
        loader_mode=args.loader_mode,
    )


if __name__ == "__main__":
    main()