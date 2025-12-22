import os
import json
import shutil
import argparse
from typing import List, Tuple
from tqdm import tqdm

def polygon_to_point_in_file(json_path: str, backup: bool = True, method: str = "centroid") -> bool:
    """
    将单个 LabelMe JSON 文件中所有 shape_type == 'polygon' 的项替换为 'point'。
    method: 'centroid' 或 'first'
    返回是否有修改。
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON: {e}")

    changed = False
    shapes = data.get("shapes", [])
    for shape in shapes:
        if shape.get("shape_type") == "polygon":
            pts = shape.get("points", []) or []
            if not pts:
                # 空多边形跳过
                continue
            if method == "first":
                new_pt = pts[0]
            else:  # centroid
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                new_pt = [sum(xs) / len(xs), sum(ys) / len(ys)]
            shape["shape_type"] = "point"
            shape["points"] = [new_pt]
            changed = True

    if changed:
        if backup:
            shutil.copy2(json_path, json_path + ".bak")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return changed


def gather_json_files(root: str, recursive: bool = False) -> List[str]:
    files = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith('.json'):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith('.json'):
                files.append(os.path.join(root, fn))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Replace LabelMe polygon shapes with point in JSON files.")
    parser.add_argument('-p',"--path", help="File or directory to process (if directory, will scan .json files).")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively search .json files under directory.")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backups.")
    parser.add_argument("--method", choices=["centroid", "first"], default="centroid", help="How to pick the replacement point.")
    parser.add_argument("--dry-run", action="store_true", help="Show which files would change but do not modify files.")
    args = parser.parse_args()

    target = args.path
    if os.path.isfile(target):
        files = [target] if target.lower().endswith('.json') else []
    elif os.path.isdir(target):
        files = gather_json_files(target, recursive=args.recursive)
    else:
        print(f"Path not found: {target}")
        return

    if not files:
        print("No .json files found to process.")
        return

    changed_files = []
    errors: List[Tuple[str, str]] = []
    for fpath in tqdm(files):
        try:
            if args.dry_run:
                # 仅检测是否会修改
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                will_change = any(s.get("shape_type") == "polygon" for s in data.get("shapes", []))
                if will_change:
                    changed_files.append(fpath)
            else:
                if polygon_to_point_in_file(fpath, backup=not args.no_backup, method=args.method):
                    changed_files.append(fpath)
        except Exception as e:
            errors.append((fpath, str(e)))

    print(f"Processed {len(files)} files. Modified: {len(changed_files)}")
    if changed_files:
        print("Modified files:")
        for p in changed_files:
            print("  " + p)
    if errors:
        print("Errors:")
        for p, msg in errors:
            print(f"  {p}: {msg}")


if __name__ == "__main__":
    main()