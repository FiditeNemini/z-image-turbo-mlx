#!/usr/bin/env python3
"""
Scan LoRA safetensors in models/loras (or a subfolder) and report failures.

Usage:
  python debugging/check_loras_import.py            # scan all loras recursively
  python debugging/check_loras_import.py --subfolder VintageBeauties
  python debugging/check_loras_import.py --subfolder "characters/" --output report.json
"""
from pathlib import Path
import argparse
import json
import traceback
import sys

# Ensure project root is on sys.path so `from src import lora` works when
# running this script directly (e.g., `python debugging/check_loras_import.py`).
# The script lives in <project>/debugging/, so parent parent is the project root.
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from src import lora as lora_mod


def scan_loras(subfolder: str | None = None):
    base = Path(lora_mod.LORAS_DIR)
    if subfolder:
        base = base / subfolder

    if not base.exists():
        raise FileNotFoundError(f"LoRA folder not found: {base}")

    files = sorted([p for p in base.rglob("*.safetensors") if p.name != ".gitkeep"])

    results = []

    for f in files:
        entry = {"path": str(f), "ok": False, "error": None, "info": None}
        try:
            # Try to get basic info (this will load weights but not apply them)
            info = lora_mod.get_lora_info(f)
            entry["ok"] = True
            entry["info"] = {
                "name": info.get("name"),
                "rank": info.get("rank"),
                "num_weights": info.get("num_weights"),
                "target_types": info.get("target_types"),
                "layer_range": info.get("layer_range"),
            }
        except Exception as e:
            entry["ok"] = False
            entry["error"] = str(e)
            entry["traceback"] = traceback.format_exc()

        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description="Check LoRA .safetensors importability")
    parser.add_argument("--subfolder", "-s", help="Subfolder under models/loras to scan (optional)")
    parser.add_argument("--output", "-o", help="Write JSON report to this path (optional)")

    args = parser.parse_args()

    try:
        results = scan_loras(args.subfolder)
    except Exception as e:
        print(f"Error: {e}")
        return 2

    total = len(results)
    failed = [r for r in results if not r["ok"]]
    ok = [r for r in results if r["ok"]]

    print(f"Scanned {total} LoRA files")
    print(f"  OK:     {len(ok)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\nFailed files:")
        for r in failed:
            print(f"- {r['path']}: {r['error']}")

    if args.output:
        outp = Path(args.output)
        outp.write_text(json.dumps({"summary": {"total": total, "ok": len(ok), "failed": len(failed)}, "results": results}, indent=2))
        print(f"\nWrote JSON report to: {outp}")

    # exit code 0 if all ok, 1 if any failed
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
