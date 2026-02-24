#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from pathlib import Path

import h5py


def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v


def copy_group_trim(src_group, dst_group, total_frames, cut_frames):
    copy_attrs(src_group, dst_group)

    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            new_group = dst_group.create_group(key)
            copy_group_trim(item, new_group, total_frames, cut_frames)
            continue

        if item.shape and item.shape[0] == total_frames:
            data = item[cut_frames:]
        else:
            data = item[()]

        kwargs = {}
        if item.compression is not None:
            kwargs["compression"] = item.compression
        if item.compression_opts is not None:
            kwargs["compression_opts"] = item.compression_opts

        new_ds = dst_group.create_dataset(key, data=data, dtype=item.dtype, **kwargs)
        copy_attrs(item, new_ds)


def get_total_frames(src_file):
    if "timestamps" in src_file:
        return int(src_file["timestamps"].shape[0])
    if "num_frames" in src_file.attrs:
        return int(src_file.attrs["num_frames"])
    raise KeyError("Cannot find frame length: missing dataset 'timestamps' and attr 'num_frames'.")


def trim_one_file(src_path, dst_path, ratio, frames):
    with h5py.File(src_path, "r") as src:
        total_frames = get_total_frames(src)
        if frames is not None:
            cut_frames = int(frames)
        else:
            cut_frames = int(total_frames * ratio)
        cut_frames = max(0, cut_frames)

        if cut_frames >= total_frames:
            raise ValueError(f"cut_frames={cut_frames} >= total_frames={total_frames}, file={src_path}")

        with h5py.File(dst_path, "w") as dst:
            copy_group_trim(src, dst, total_frames, cut_frames)
            if "num_frames" in dst.attrs:
                dst.attrs["num_frames"] = total_frames - cut_frames

    return total_frames, cut_frames


def main():
    parser = argparse.ArgumentParser(description="Trim the first ratio of frames from episode_*.hdf5 files.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data_gaze_human_120",
        help="Folder containing episode_*.hdf5",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.10,
        help="Head ratio to remove, e.g. 0.10 means first 10%% frames",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=-1,
        help="If >= 0, remove a fixed number of head frames (higher priority than --ratio)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original files in input_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output folder when not using --inplace (default: <input_dir>_trimmed)",
    )
    args = parser.parse_args()

    if not (0.0 <= args.ratio < 1.0):
        raise ValueError("--ratio must be in [0.0, 1.0).")
    if args.frames < -1:
        raise ValueError("--frames must be >= -1.")
    fixed_frames = args.frames if args.frames >= 0 else None

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    files = sorted(glob.glob(str(input_dir / "episode_*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 found in {input_dir}")

    if args.inplace:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"{input_dir}_trimmed")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] files={len(files)}, ratio={args.ratio}, frames={fixed_frames}, inplace={args.inplace}"
    )
    print(f"[INFO] input_dir={input_dir}")
    print(f"[INFO] output_dir={output_dir}")

    for src in files:
        src_path = Path(src)
        if args.inplace:
            tmp_path = src_path.with_suffix(".tmp.hdf5")
            total, cut = trim_one_file(src_path, tmp_path, args.ratio, fixed_frames)
            os.replace(tmp_path, src_path)
            dst_path = src_path
        else:
            dst_path = output_dir / src_path.name
            total, cut = trim_one_file(src_path, dst_path, args.ratio, fixed_frames)

        print(f"[OK] {src_path.name}: {total} -> {total - cut} (removed {cut})")

    print("[DONE]")


if __name__ == "__main__":
    main()
