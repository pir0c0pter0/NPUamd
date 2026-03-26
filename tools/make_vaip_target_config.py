#!/usr/bin/env python3
"""Create a derived VAIP config with a forced target."""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Create derived VAIP config with a forced target")
    parser.add_argument("base_config", help="Path to base VAIP JSON config")
    parser.add_argument("output_config", help="Path to derived config to write")
    parser.add_argument("--target", required=True, help="Target name to force")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    target_names = {t.get("name") for t in config.get("targets", [])}
    if args.target not in target_names:
        print(f"[ERROR] target not found in base config: {args.target}", file=sys.stderr)
        print(f"[INFO] available targets: {sorted(target_names)}", file=sys.stderr)
        return 1

    config["target"] = args.target

    out_dir = os.path.dirname(os.path.abspath(args.output_config))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    print(f"[OK] wrote {args.output_config}")
    print(f"[OK] forced target: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
