"""
Dataset management for Z-Image-Turbo training.

This module provides:
- Dataset directory structure management
- Image preprocessing and augmentation
- Caption loading and processing
- Latent caching for efficient training
- Aspect ratio bucketing

Dataset Structure:
    datasets/
    └── my_dataset/
        ├── dataset.json       # Dataset configuration
        └── images/
            ├── image1.png
            ├── image1.txt     # Caption file (same name as image)
            ├── image2.jpg
            └── image2.txt
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
import json
import random
import math
from dataclasses import dataclass, field

# Optional torchvision import
try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    transforms = None


# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Default datasets directory
DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"


@dataclass
class ImageEntry:
    """Represents a single image in the dataset."""
    
    image_path: Path
    caption: str = ""
    width: int = 0
    height: int = 0
    latent_path: Optional[Path] = None
    text_embed_path: Optional[Path] = None
    
    def __post_init__(self):
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)


@dataclass
class DatasetMetadata:
    """Metadata for a training dataset."""
    
    name: str = ""
    description: str = ""
    num_images: int = 0
    trigger_word: str = ""
    default_caption: str = ""
    resolution: int = 1024
    created_at: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "num_images": self.num_images,
            "trigger_word": self.trigger_word,
            "default_caption": self.default_caption,
            "resolution": self.resolution,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DatasetManager:
    """
    Manages training datasets including creation, loading, and validation.
    """
    
    def __init__(self, datasets_dir: Optional[Path] = None):
        """
        Initialize dataset manager.
        
        Args:
            datasets_dir: Root directory for datasets
        """
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_DIR
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self) -> List[str]:
        """
        List available datasets.
        
        Returns:
            List of dataset names
        """
        datasets = []
        for d in self.datasets_dir.iterdir():
            if d.is_dir() and (d / "images").exists():
                datasets.append(d.name)
        return sorted(datasets)
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            name: Dataset name
            
        Returns:
            Dictionary with dataset information
        """
        dataset_path = self.datasets_dir / name
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {name}")
        
        images_dir = dataset_path / "images"
        
        # Load metadata if exists
        metadata_path = dataset_path / "dataset.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Count images
        image_files = self._find_image_files(images_dir)
        
        # Get image dimensions if any exist
        dimensions = []
        for img_path in image_files[:10]:  # Sample first 10
            try:
                with Image.open(img_path) as img:
                    dimensions.append((img.width, img.height))
            except Exception:
                pass
        
        return {
            "name": name,
            "path": str(dataset_path),
            "num_images": len(image_files),
            "sample_dimensions": dimensions,
            "has_captions": self._count_captions(images_dir, image_files) > 0,
            "num_captions": self._count_captions(images_dir, image_files),
            "metadata": metadata,
        }
    
    def create_dataset(
        self,
        name: str,
        description: str = "",
        trigger_word: str = "",
        default_caption: str = "",
        resolution: int = 1024,
    ) -> Path:
        """
        Create a new dataset directory structure.
        
        Args:
            name: Dataset name (will be used as directory name)
            description: Description of the dataset
            trigger_word: Trigger word for the concept being trained
            default_caption: Default caption template
            resolution: Target training resolution
            
        Returns:
            Path to the created dataset directory
        """
        from datetime import datetime
        
        # Sanitize name
        name = "".join(c for c in name if c.isalnum() or c in "._- ")
        name = name.strip()
        
        if not name:
            raise ValueError("Dataset name cannot be empty")
        
        dataset_path = self.datasets_dir / name
        images_dir = dataset_path / "images"
        
        if dataset_path.exists():
            raise ValueError(f"Dataset already exists: {name}")
        
        # Create directories
        images_dir.mkdir(parents=True)
        
        # Create metadata file
        metadata = DatasetMetadata(
            name=name,
            description=description,
            trigger_word=trigger_word,
            default_caption=default_caption,
            resolution=resolution,
            created_at=datetime.now().isoformat(),
        )
        
        with open(dataset_path / "dataset.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return dataset_path
    
    def add_images(
        self,
        dataset_name: str,
        image_paths: List[Union[str, Path]],
        captions: Optional[List[str]] = None,
        copy_files: bool = True,
    ) -> int:
        """
        Add images to a dataset.
        
        Args:
            dataset_name: Name of the dataset
            image_paths: List of paths to images
            captions: Optional list of captions (same length as image_paths)
            copy_files: If True, copy files; if False, move them
            
        Returns:
            Number of images added
        """
        import shutil
        
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        images_dir = dataset_path / "images"
        
        if captions is not None and len(captions) != len(image_paths):
            raise ValueError("Number of captions must match number of images")
        
        added_count = 0
        
        for i, src_path in enumerate(image_paths):
            src_path = Path(src_path)
            
            if not src_path.exists():
                print(f"Warning: Image not found: {src_path}")
                continue
            
            if src_path.suffix.lower() not in SUPPORTED_FORMATS:
                print(f"Warning: Unsupported format: {src_path}")
                continue
            
            # Destination path
            dst_path = images_dir / src_path.name
            
            # Handle name conflicts
            if dst_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while dst_path.exists():
                    dst_path = images_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            # Copy or move file
            if copy_files:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)
            
            # Write caption if provided
            if captions is not None and captions[i]:
                caption_path = dst_path.with_suffix(".txt")
                with open(caption_path, "w") as f:
                    f.write(captions[i])
            
            added_count += 1
        
        # Update metadata
        self._update_metadata_count(dataset_name)
        
        return added_count
    
    def load_dataset(self, dataset_name: str) -> List[ImageEntry]:
        """
        Load all images and captions from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of ImageEntry objects
        """
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        images_dir = dataset_path / "images"
        
        # Load metadata for default caption
        metadata_path = dataset_path / "dataset.json"
        default_caption = ""
        trigger_word = ""
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                default_caption = metadata.get("default_caption", "")
                trigger_word = metadata.get("trigger_word", "")
        
        entries = []
        image_files = self._find_image_files(images_dir)
        
        for img_path in image_files:
            # Load caption
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                with open(caption_path) as f:
                    caption = f.read().strip()
            else:
                caption = default_caption
            
            # Add trigger word if present
            if trigger_word and trigger_word not in caption:
                caption = f"{trigger_word}, {caption}" if caption else trigger_word
            
            # Get image dimensions
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0
            
            entries.append(ImageEntry(
                image_path=img_path,
                caption=caption,
                width=width,
                height=height,
            ))
        
        return entries
    
    def validate_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate a dataset and report any issues.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Validation report dictionary
        """
        dataset_path = self.datasets_dir / dataset_name
        if not dataset_path.exists():
            return {"valid": False, "error": f"Dataset not found: {dataset_name}"}
        
        images_dir = dataset_path / "images"
        if not images_dir.exists():
            return {"valid": False, "error": "No images directory found"}
        
        image_files = self._find_image_files(images_dir)
        
        if not image_files:
            return {"valid": False, "error": "No images found in dataset"}
        
        issues = []
        warnings = []
        
        # Check each image
        valid_images = 0
        images_with_captions = 0
        
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_images += 1
                
                # Check for caption
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    images_with_captions += 1
                else:
                    warnings.append(f"No caption for: {img_path.name}")
                    
            except Exception as e:
                issues.append(f"Invalid image {img_path.name}: {e}")
        
        # Summary
        caption_ratio = images_with_captions / len(image_files) if image_files else 0
        
        if caption_ratio < 1.0:
            warnings.append(f"Only {images_with_captions}/{len(image_files)} images have captions")
        
        return {
            "valid": len(issues) == 0,
            "total_images": len(image_files),
            "valid_images": valid_images,
            "images_with_captions": images_with_captions,
            "caption_coverage": round(caption_ratio * 100, 1),
            "issues": issues,
            "warnings": warnings,
        }
    
    def _find_image_files(self, images_dir: Path) -> List[Path]:
        """Find all image files in directory."""
        files = []
        for f in images_dir.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
                files.append(f)
        return sorted(files)
    
    def _count_captions(self, images_dir: Path, image_files: List[Path]) -> int:
        """Count images with caption files."""
        count = 0
        for img_path in image_files:
            if img_path.with_suffix(".txt").exists():
                count += 1
        return count
    
    def _update_metadata_count(self, dataset_name: str):
        """Update image count in metadata file."""
        dataset_path = self.datasets_dir / dataset_name
        metadata_path = dataset_path / "dataset.json"
        
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        images_dir = dataset_path / "images"
        image_files = self._find_image_files(images_dir)
        metadata["num_images"] = len(image_files)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


class TrainingDataset(Dataset):
    """
    PyTorch Dataset for Z-Image-Turbo training.
    
    Handles image loading, preprocessing, and optional latent caching.
    Supports both fixed resolution and bucketed (variable resolution) modes.
    """
    
    def __init__(
        self,
        entries: List[ImageEntry],
        resolution: int = 1024,
        transform: Optional[Callable] = None,
        flip_horizontal: bool = True,
        flip_vertical: bool = False,
        random_crop: bool = False,
        cached_latents: Optional[Dict[str, torch.Tensor]] = None,
        cached_text_embeds: Optional[Dict[str, torch.Tensor]] = None,
        bucket_manager: Optional["AspectRatioBucket"] = None,
    ):
        """
        Initialize training dataset.
        
        Args:
            entries: List of ImageEntry objects
            resolution: Target resolution (used when not bucketing)
            transform: Optional custom transform
            flip_horizontal: Enable horizontal flipping
            flip_vertical: Enable vertical flipping
            random_crop: Use random crop instead of center crop
            cached_latents: Pre-computed latents (keyed by image path)
            cached_text_embeds: Pre-computed text embeddings (keyed by caption hash)
            bucket_manager: Optional AspectRatioBucket for variable resolutions
        """
        self.entries = entries
        self.resolution = resolution
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.random_crop = random_crop
        self.cached_latents = cached_latents or {}
        self.cached_text_embeds = cached_text_embeds or {}
        self.bucket_manager = bucket_manager
        
        # Pre-compute bucket assignments if bucketing is enabled
        self.entry_buckets: Dict[int, Tuple[int, int]] = {}
        if bucket_manager is not None:
            for idx, entry in enumerate(entries):
                if entry.width > 0 and entry.height > 0:
                    self.entry_buckets[idx] = bucket_manager.get_bucket(entry.width, entry.height)
                else:
                    self.entry_buckets[idx] = (resolution, resolution)
        
        # Only create default transform if not bucketing (bucketing creates per-sample transforms)
        if transform is not None:
            self.transform = transform
        elif bucket_manager is None:
            self.transform = self._create_transform(resolution, resolution)
        else:
            self.transform = None  # Will create per-sample in __getitem__
    
    def _create_transform(self, target_width: int, target_height: int) -> Callable:
        """Create image transform pipeline for a specific target size."""
        if TORCHVISION_AVAILABLE:
            # For bucketing, we resize to fit the bucket then crop to exact size
            transform_list = []
            
            if target_width == target_height:
                # Square: resize shortest side, then crop
                transform_list.append(
                    transforms.Resize(
                        target_width,
                        interpolation=transforms.InterpolationMode.LANCZOS,
                    )
                )
                if self.random_crop:
                    transform_list.append(transforms.RandomCrop(target_width))
                else:
                    transform_list.append(transforms.CenterCrop(target_width))
            else:
                # Non-square: resize to fit bucket dimensions
                transform_list.append(
                    transforms.Resize(
                        (target_height, target_width),
                        interpolation=transforms.InterpolationMode.LANCZOS,
                    )
                )
            
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
            ])
            
            return transforms.Compose(transform_list)
        else:
            # Fallback without torchvision - create a closure
            return lambda img: self._pil_transform(img, target_width, target_height)
    
    def _pil_transform(self, image: Image.Image, target_width: int, target_height: int) -> torch.Tensor:
        """Fallback transform using PIL only (when torchvision not available)."""
        import numpy as np
        
        w, h = image.size
        
        if target_width == target_height:
            # Square: resize maintaining aspect ratio then center crop
            scale = target_width / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Center crop to square
            left = (new_w - target_width) // 2
            top = (new_h - target_height) // 2
            image = image.crop((left, top, left + target_width, top + target_height))
        else:
            # Non-square bucket: resize to exact dimensions
            image = image.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        arr = np.array(image).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
            - pixel_values: Image tensor [C, H, W]
            - caption: Caption string
            - image_path: Path to original image
            - bucket_size: (width, height) tuple if bucketing
            - latent: Cached latent if available
            - text_embed: Cached text embedding if available
        """
        entry = self.entries[idx]
        
        # Load and transform image
        image = Image.open(entry.image_path).convert("RGB")
        
        # Apply random flips
        if self.flip_horizontal and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.flip_vertical and random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Get target size (from bucket or default resolution)
        if self.bucket_manager is not None and idx in self.entry_buckets:
            target_w, target_h = self.entry_buckets[idx]
            transform = self._create_transform(target_w, target_h)
            pixel_values = transform(image)
        else:
            target_w, target_h = self.resolution, self.resolution
            pixel_values = self.transform(image)
        
        sample = {
            "pixel_values": pixel_values,
            "caption": entry.caption,
            "image_path": str(entry.image_path),
            "bucket_size": (target_w, target_h),
        }
        
        # Add cached latent if available
        path_key = str(entry.image_path)
        if path_key in self.cached_latents:
            sample["latent"] = self.cached_latents[path_key]
        
        # Add cached text embedding if available
        caption_key = hash(entry.caption)
        if caption_key in self.cached_text_embeds:
            sample["text_embed"] = self.cached_text_embeds[caption_key]
        
        return sample


class AspectRatioBucket:
    """
    Manages aspect ratio bucketing for efficient training.
    
    Groups images by similar aspect ratios to minimize padding waste.
    """
    
    def __init__(
        self,
        base_resolution: int = 1024,
        max_resolution: int = 1024,
        min_resolution: int = 512,
        step_size: int = 64,
        max_aspect_ratio: float = 2.0,
    ):
        """
        Initialize bucket manager.
        
        Args:
            base_resolution: Base resolution (used for total pixel count)
            max_resolution: Maximum dimension
            min_resolution: Minimum dimension  
            step_size: Resolution step size (for creating buckets)
            max_aspect_ratio: Maximum allowed aspect ratio
        """
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.min_resolution = min_resolution
        self.step_size = step_size
        self.max_aspect_ratio = max_aspect_ratio
        
        # Generate bucket sizes
        self.buckets = self._generate_buckets()
    
    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """Generate all valid bucket sizes maintaining roughly constant total pixels."""
        buckets = set()
        
        # Target total pixels (from base resolution)
        target_pixels = self.base_resolution ** 2
        
        # Generate bucket sizes - iterate through possible widths
        # and calculate corresponding height to maintain similar pixel count
        for width in range(self.min_resolution, self.max_resolution + 1, self.step_size):
            # Calculate height to maintain ~same total pixels
            height = int(target_pixels / width)
            # Round to step size
            height = (height // self.step_size) * self.step_size
            
            # Skip if height is out of valid range
            if height < self.min_resolution:
                continue
            
            # Cap height at a reasonable maximum (allow some flexibility)
            max_h = int(self.base_resolution * self.max_aspect_ratio)
            max_h = (max_h // self.step_size) * self.step_size
            if height > max_h:
                height = max_h
            
            # Check aspect ratio
            aspect = max(width / height, height / width)
            if aspect > self.max_aspect_ratio:
                continue
            
            buckets.add((width, height))
            # Also add rotated version (portrait)
            if width != height:
                buckets.add((height, width))
        
        # Always include square bucket
        buckets.add((self.base_resolution, self.base_resolution))
        
        return sorted(list(buckets))
    
    def get_bucket(self, width: int, height: int) -> Tuple[int, int]:
        """
        Get the best bucket for an image.
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            (bucket_width, bucket_height) tuple
        """
        aspect = width / height
        
        # Find closest bucket by aspect ratio
        best_bucket = None
        best_diff = float("inf")
        
        for bw, bh in self.buckets:
            bucket_aspect = bw / bh
            diff = abs(aspect - bucket_aspect)
            
            if diff < best_diff:
                best_diff = diff
                best_bucket = (bw, bh)
        
        return best_bucket or (self.base_resolution, self.base_resolution)
    
    def assign_buckets(self, entries: List[ImageEntry]) -> Dict[Tuple[int, int], List[ImageEntry]]:
        """
        Assign images to buckets.
        
        Args:
            entries: List of ImageEntry objects
            
        Returns:
            Dictionary mapping bucket sizes to lists of entries
        """
        bucket_assignments: Dict[Tuple[int, int], List[ImageEntry]] = {}
        
        for entry in entries:
            if entry.width > 0 and entry.height > 0:
                bucket = self.get_bucket(entry.width, entry.height)
            else:
                bucket = (self.base_resolution, self.base_resolution)
            
            if bucket not in bucket_assignments:
                bucket_assignments[bucket] = []
            bucket_assignments[bucket].append(entry)
        
        return bucket_assignments
    
    def get_bucket_info(self) -> str:
        """Get human-readable bucket information."""
        lines = [f"Aspect Ratio Buckets (base {self.base_resolution}px):"]
        for w, h in sorted(self.buckets, key=lambda x: x[0]/x[1]):
            aspect = w / h
            lines.append(f"  {w}x{h} (aspect {aspect:.2f})")
        return "\n".join(lines)


class BucketBatchSampler:
    """
    Batch sampler that groups images by their aspect ratio bucket.
    
    This ensures all images in a batch have the same target dimensions,
    allowing efficient batching without padding waste.
    """
    
    def __init__(
        self,
        dataset: TrainingDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        """
        Initialize bucket batch sampler.
        
        Args:
            dataset: TrainingDataset with bucket assignments
            batch_size: Number of samples per batch
            drop_last: Drop incomplete batches
            shuffle: Shuffle within buckets each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Group indices by bucket
        self.bucket_indices: Dict[Tuple[int, int], List[int]] = {}
        for idx in range(len(dataset)):
            bucket = dataset.entry_buckets.get(idx, (dataset.resolution, dataset.resolution))
            if bucket not in self.bucket_indices:
                self.bucket_indices[bucket] = []
            self.bucket_indices[bucket].append(idx)
        
        # Calculate total batches
        self._calculate_batches()
    
    def _calculate_batches(self):
        """Calculate batches for current epoch."""
        self.batches = []
        
        for bucket, indices in self.bucket_indices.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            # Create batches from this bucket
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.batches.append(batch)
        
        # Shuffle batches across buckets
        if self.shuffle:
            random.shuffle(self.batches)
    
    def __iter__(self):
        self._calculate_batches()  # Re-shuffle each epoch
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)
    
    def get_bucket_stats(self) -> Dict[str, Any]:
        """Get statistics about bucket distribution."""
        stats = {
            "num_buckets": len(self.bucket_indices),
            "buckets": {},
        }
        for bucket, indices in self.bucket_indices.items():
            stats["buckets"][f"{bucket[0]}x{bucket[1]}"] = {
                "count": len(indices),
                "batches": len(indices) // self.batch_size,
            }
        return stats


def create_dataloader(
    dataset: TrainingDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training (fixed resolution, no bucketing).
    
    Args:
        dataset: Training dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches
        collate_fn=_collate_fn,
    )


def create_bucketed_dataloader(
    dataset: TrainingDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, BucketBatchSampler]:
    """
    Create a DataLoader with aspect ratio bucketing.
    
    Images are grouped by similar aspect ratios so each batch contains
    images with the same target dimensions, eliminating padding waste.
    
    Args:
        dataset: Training dataset with bucket_manager set
        batch_size: Batch size
        shuffle: Whether to shuffle within and across buckets
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (DataLoader, BucketBatchSampler)
    """
    sampler = BucketBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=shuffle,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )
    
    return dataloader, sampler


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching training samples.
    """
    result = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }
    
    # Include bucket size info
    if "bucket_size" in batch[0]:
        result["bucket_size"] = batch[0]["bucket_size"]  # All same in bucketed batch
    
    # Handle optional cached values
    if "latent" in batch[0]:
        result["latents"] = torch.stack([b["latent"] for b in batch])
    
    if "text_embed" in batch[0]:
        result["text_embeds"] = torch.stack([b["text_embed"] for b in batch])
    
    return result
