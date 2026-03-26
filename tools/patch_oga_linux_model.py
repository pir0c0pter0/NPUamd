#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def is_lfs_pointer(path: Path) -> bool:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (IndexError, UnicodeDecodeError):
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        fail(f"missing {label}: {path}")
    if path.stat().st_size == 0:
        fail(f"{label} is empty: {path}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def patch_genai_config(config_path: Path, custom_ops_library: str, aie_rope_flag: str) -> None:
    require_nonempty_file(config_path, "genai_config.json")
    if is_lfs_pointer(config_path):
        fail(f"genai_config.json is still a git-lfs pointer: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    session_options = config.setdefault("model", {}).setdefault("decoder", {}).setdefault("session_options", {})
    session_options["custom_ops_library"] = custom_ops_library
    config_entries = session_options.setdefault("config_entries", {})
    config_entries["hybrid_dbg_use_aie_rope"] = aie_rope_flag

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
        handle.write("\n")


def patch_meta_json(meta_path: Path) -> None:
    require_nonempty_file(meta_path, "OGA meta file")
    if is_lfs_pointer(meta_path):
        fail(f"OGA meta file is still a git-lfs pointer: {meta_path}")

    lines = meta_path.read_text(encoding="utf-8").splitlines(keepends=True)
    patched = []

    for line in lines:
        if ".cache" in line:
            patched.append(line.replace("\\", "/"))
        else:
            patched.append(line)

    meta_path.write_text("".join(patched), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch an AMD OGA model tree to match the official Linux LLM flow."
    )
    parser.add_argument("model_dir", help="Path to the extracted model directory")
    parser.add_argument(
        "--custom-ops-library",
        default="deployment/lib/libonnx_custom_ops.so",
        help="Path written into genai_config.json for the Linux custom ops library",
    )
    parser.add_argument(
        "--aie-rope-flag",
        default="0",
        help="Value for config_entries.hybrid_dbg_use_aie_rope",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    config_path = model_dir / "genai_config.json"
    meta_path = model_dir / ".cache" / "MatMulNBits_2_0_meta.json"

    if not model_dir.is_dir():
        fail(f"missing model directory: {model_dir}")

    patch_genai_config(config_path, args.custom_ops_library, args.aie_rope_flag)
    patch_meta_json(meta_path)

    print(f"[INFO] patched Linux OGA config: {config_path}")
    print(f"[INFO] patched cache metadata paths: {meta_path}")


if __name__ == "__main__":
    main()
