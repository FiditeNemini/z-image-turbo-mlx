#!/usr/bin/env python3
"""
Migration script for Z-Image-Turbo MLX model directory structure.

This script migrates models from the old flat structure to the new organized structure:

Old structure:
  models/
  ├── Z-Image-Turbo/        (PyTorch model)
  ├── mlx_model/            (MLX model)
  └── prompt_enhancer/      (Prompt enhancement model)

New structure:
  models/
  ├── mlx/
  │   └── mlx_model/        (MLX models go here)
  ├── pytorch/
  │   └── Z-Image-Turbo/    (PyTorch models go here)
  └── prompt_enhancer/      (Unchanged - utility model)

Usage:
  python migrate_models.py           # Preview changes (dry run)
  python migrate_models.py --apply   # Apply the migration
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Z-Image-Turbo models to new directory structure"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration (without this flag, only shows what would be done)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Path to models directory (default: ./models)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    dry_run = not args.apply

    if dry_run:
        print("=" * 60)
        print("DRY RUN - No changes will be made")
        print("Run with --apply to perform the migration")
        print("=" * 60)
        print()

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        print("   Nothing to migrate.")
        return

    # Define new directories
    mlx_dir = models_dir / "mlx"
    pytorch_dir = models_dir / "pytorch"

    # Check if already migrated
    if mlx_dir.exists() and pytorch_dir.exists():
        print("✓ Directory structure appears to already be migrated.")
        print(f"  MLX models: {mlx_dir}")
        print(f"  PyTorch models: {pytorch_dir}")
        
        # List contents
        if mlx_dir.exists():
            mlx_models = [d.name for d in mlx_dir.iterdir() if d.is_dir()]
            print(f"\n  MLX models found: {mlx_models if mlx_models else '(none)'}")
        if pytorch_dir.exists():
            pytorch_models = [d.name for d in pytorch_dir.iterdir() if d.is_dir()]
            print(f"  PyTorch models found: {pytorch_models if pytorch_models else '(none)'}")
        return

    print("Scanning for models to migrate...\n")

    migrations = []
    skipped = []

    # Check for old-style models in the root models directory
    for item in models_dir.iterdir():
        if not item.is_dir():
            continue

        # Skip if it's already the new structure directories
        if item.name in ["mlx", "pytorch", "unsupported"]:
            continue

        # Skip prompt_enhancer - it stays where it is
        if item.name == "prompt_enhancer":
            skipped.append((item, "Utility model - stays in place"))
            continue

        # Detect model type
        is_mlx = False
        is_pytorch = False

        # Check for MLX model signatures
        if (item / "weights.safetensors").exists() and (item / "config.json").exists():
            # Could be MLX - check if it has MLX-style weights
            is_mlx = True

        # Check for PyTorch/diffusers model signatures
        if (item / "transformer").exists() or (item / "model_index.json").exists():
            is_pytorch = True

        # Check for diffusers sharded model
        transformer_dir = item / "transformer"
        if transformer_dir.exists():
            if list(transformer_dir.glob("diffusion_pytorch_model*.safetensors")):
                is_pytorch = True
                is_mlx = False  # Override - this is definitely PyTorch

        # Determine destination
        if is_pytorch and not is_mlx:
            dest = pytorch_dir / item.name
            migrations.append((item, dest, "PyTorch (diffusers format)"))
        elif is_mlx and not is_pytorch:
            dest = mlx_dir / item.name
            migrations.append((item, dest, "MLX"))
        elif is_mlx and is_pytorch:
            # Has both - probably MLX (converted from PyTorch)
            dest = mlx_dir / item.name
            migrations.append((item, dest, "MLX (with config)"))
        else:
            skipped.append((item, "Unknown format - manual review needed"))

    # Print migration plan
    if migrations:
        print("📦 Models to migrate:")
        print("-" * 60)
        for src, dest, model_type in migrations:
            print(f"  {src.name}")
            print(f"    Type: {model_type}")
            print(f"    From: {src}")
            print(f"    To:   {dest}")
            print()

    if skipped:
        print("⏭️  Skipped:")
        print("-" * 60)
        for item, reason in skipped:
            print(f"  {item.name}: {reason}")
        print()

    if not migrations:
        print("✓ No models need to be migrated.")
        return

    # Perform migration
    if dry_run:
        print("-" * 60)
        print("To apply these changes, run:")
        print(f"  python {__file__} --apply")
    else:
        print("-" * 60)
        print("Applying migration...")
        print()

        # Create directories
        mlx_dir.mkdir(parents=True, exist_ok=True)
        pytorch_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {mlx_dir}")
        print(f"✓ Created {pytorch_dir}")

        # Move models
        for src, dest, model_type in migrations:
            try:
                if dest.exists():
                    print(f"⚠️  Destination already exists, skipping: {dest}")
                    continue
                shutil.move(str(src), str(dest))
                print(f"✓ Moved {src.name} → {dest}")
            except Exception as e:
                print(f"❌ Error moving {src.name}: {e}")

        print()
        print("=" * 60)
        print("✅ Migration complete!")
        print("=" * 60)
        print()
        print("New structure:")
        print(f"  {mlx_dir}/")
        for item in mlx_dir.iterdir():
            if item.is_dir():
                print(f"    └── {item.name}/")
        print(f"  {pytorch_dir}/")
        for item in pytorch_dir.iterdir():
            if item.is_dir():
                print(f"    └── {item.name}/")


if __name__ == "__main__":
    main()
