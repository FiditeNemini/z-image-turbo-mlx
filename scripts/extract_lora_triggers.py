#!/usr/bin/env python3
"""
Standalone script to extract trigger words from LoRA safetensors files.

This script reads the metadata embedded in a LoRA .safetensors file and
extracts trigger words from common metadata fields.

Usage:
    python extract_lora_triggers.py path/to/lora.safetensors
    python extract_lora_triggers.py path/to/loras/folder/

Requirements:
    pip install safetensors

Trigger words are typically stored in these metadata fields:
    - ss_tag_frequency: JSON dict of {folder: {tag: count}} (kohya_ss/ai-toolkit format)
    - trigger_word / trigger_words: Single string or comma-separated list
    - activation_text: Alternative name used by some trainers

Additionally checks for a companion .txt file with the same name as the LoRA.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def get_safetensors_metadata(filepath: Path) -> Dict[str, str]:
    """
    Read metadata from a safetensors file header.
    
    Args:
        filepath: Path to the .safetensors file
        
    Returns:
        Dictionary of metadata key-value pairs
    """
    from safetensors import safe_open
    
    metadata = {}
    try:
        with safe_open(str(filepath), framework="numpy", device="cpu") as f:
            raw_metadata = f.metadata()
            if raw_metadata:
                metadata = dict(raw_metadata)
    except Exception as e:
        print(f"Error reading metadata: {e}", file=sys.stderr)
    
    return metadata


def extract_trigger_words(lora_path: Path) -> List[str]:
    """
    Extract trigger words from a LoRA file's metadata.
    
    Checks multiple common metadata fields and also looks for
    a companion .txt file with the same name.
    
    Args:
        lora_path: Path to the LoRA .safetensors file
        
    Returns:
        List of unique trigger words found
    """
    trigger_words = []
    
    # Read safetensors metadata
    metadata = get_safetensors_metadata(lora_path)
    
    # 1. Check ss_tag_frequency (kohya_ss / ai-toolkit format)
    #    Format: {"folder_name": {"tag1": count, "tag2": count, ...}}
    if "ss_tag_frequency" in metadata:
        try:
            tag_freq = json.loads(metadata["ss_tag_frequency"])
            for folder_tags in tag_freq.values():
                if isinstance(folder_tags, dict):
                    # The keys are the tags/trigger words
                    trigger_words.extend(folder_tags.keys())
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  Warning: Could not parse ss_tag_frequency: {e}", file=sys.stderr)
    
    # 2. Check common trigger word fields
    for field in ["trigger_word", "trigger_words", "activation_text"]:
        if field in metadata:
            value = metadata[field]
            if isinstance(value, str) and value.strip():
                # Could be comma-separated
                words = [w.strip() for w in value.split(",") if w.strip()]
                trigger_words.extend(words)
    
    # 3. Check CivitAI modelspec format
    if "modelspec.metadata" in metadata:
        try:
            spec = json.loads(metadata["modelspec.metadata"])
            if "trigger_phrase" in spec:
                trigger_words.append(spec["trigger_phrase"])
            if "activation_text" in spec:
                trigger_words.append(spec["activation_text"])
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
    
    # 4. Check for companion .txt file
    txt_path = lora_path.with_suffix(".txt")
    if txt_path.exists():
        try:
            content = txt_path.read_text().strip()
            if content:
                # Could be one per line or comma-separated
                for line in content.splitlines():
                    words = [w.strip() for w in line.split(",") if w.strip()]
                    trigger_words.extend(words)
        except Exception as e:
            print(f"  Warning: Could not read {txt_path.name}: {e}", file=sys.stderr)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_words = []
    for word in trigger_words:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_words.append(word)
    
    return unique_words


def get_lora_info(lora_path: Path) -> Dict:
    """
    Get extended information about a LoRA file.
    
    Args:
        lora_path: Path to the LoRA file
        
    Returns:
        Dictionary with LoRA information
    """
    metadata = get_safetensors_metadata(lora_path)
    trigger_words = extract_trigger_words(lora_path)
    
    # Extract additional useful metadata
    info = {
        "path": str(lora_path),
        "name": lora_path.stem,
        "trigger_words": trigger_words,
        "metadata_fields": list(metadata.keys()),
    }
    
    # Common metadata fields to extract
    for field in ["ss_network_dim", "ss_network_alpha", "ss_base_model_version", 
                  "ss_training_started_at", "ss_epoch", "ss_steps"]:
        if field in metadata:
            info[field] = metadata[field]
    
    return info


def process_file(filepath: Path, verbose: bool = False) -> None:
    """Process a single LoRA file and print results."""
    print(f"\n📦 {filepath.name}")
    print(f"   Path: {filepath}")
    
    trigger_words = extract_trigger_words(filepath)
    
    if trigger_words:
        print(f"   🏷️  Trigger words:")
        for word in trigger_words:
            print(f"       • {word}")
    else:
        print("   ⚠️  No trigger words found in metadata")
    
    if verbose:
        metadata = get_safetensors_metadata(filepath)
        print(f"   📋 Metadata fields: {', '.join(metadata.keys()) or 'none'}")
        
        # Show rank if available
        if "ss_network_dim" in metadata:
            print(f"   🔢 Rank (dim): {metadata['ss_network_dim']}")
        if "ss_network_alpha" in metadata:
            print(f"   📐 Alpha: {metadata['ss_network_alpha']}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract trigger words from LoRA safetensors files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_lora_triggers.py my_lora.safetensors
    python extract_lora_triggers.py ./loras/ --verbose
    python extract_lora_triggers.py ./loras/ --json
        """
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .safetensors file or directory containing LoRAs"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional metadata information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    if not args.path.exists():
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    # Collect files to process
    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("*.safetensors"))
        if not files:
            print(f"No .safetensors files found in {args.path}", file=sys.stderr)
            sys.exit(1)
    
    # JSON output mode
    if args.json:
        results = []
        for filepath in files:
            info = get_lora_info(filepath)
            results.append(info)
        print(json.dumps(results, indent=2))
        return
    
    # Normal output mode
    print(f"🔍 Scanning {len(files)} LoRA file(s)...")
    
    for filepath in files:
        process_file(filepath, verbose=args.verbose)
    
    print()


if __name__ == "__main__":
    main()
