"""
Upscaler Download Utilities

Auto-downloads popular ESRGAN upscaler models if none are present.
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)

# Known-good ESRGAN upscalers with direct download URLs
# These are hosted on Hugging Face for reliable downloads
AVAILABLE_UPSCALERS: Dict[str, Dict[str, str]] = {
    "4x-UltraSharp": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
        "filename": "4x-UltraSharp.pth",
        "description": "General purpose, excellent quality",
        "size_mb": 64,
    },
    "4x_foolhardy_Remacri": {
        "url": "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
        "filename": "4x_foolhardy_Remacri.pth",
        "description": "Photo realistic, good for faces",
        "size_mb": 64,
    },
    "4x_NMKD-Siax_200k": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth",
        "filename": "4x_NMKD-Siax_200k.pth",
        "description": "Balanced quality and sharpness",
        "size_mb": 64,
    },
    "4x_NMKD-Superscale-SP_178000_G": {
        "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Superscale-SP_178000_G.pth",
        "filename": "4x_NMKD-Superscale-SP_178000_G.pth",
        "description": "Sharp details, good for anime/artwork",
        "size_mb": 64,
    },
}

# Default upscaler to download if none exist
DEFAULT_UPSCALER = "4x-UltraSharp"


def get_upscalers_dir() -> Path:
    """Get the upscalers directory path."""
    return Path("./models/upscalers")


def list_installed_upscalers() -> List[str]:
    """List installed upscaler files."""
    upscalers_dir = get_upscalers_dir()
    if not upscalers_dir.exists():
        return []
    return [f.stem for f in upscalers_dir.glob("*.pth")]


def is_upscaler_installed(name: str) -> bool:
    """Check if a specific upscaler is installed."""
    upscalers_dir = get_upscalers_dir()
    if name in AVAILABLE_UPSCALERS:
        filename = AVAILABLE_UPSCALERS[name]["filename"]
    else:
        filename = f"{name}.pth"
    return (upscalers_dir / filename).exists()


def download_upscaler(
    name: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Download a single upscaler model.
    
    Args:
        name: Name of the upscaler (must be in AVAILABLE_UPSCALERS)
        progress_callback: Optional callback(progress: float, message: str)
        
    Returns:
        True if successful, False otherwise
    """
    if name not in AVAILABLE_UPSCALERS:
        logger.error(f"Unknown upscaler: {name}")
        return False
    
    info = AVAILABLE_UPSCALERS[name]
    url = info["url"]
    filename = info["filename"]
    size_mb = info["size_mb"]
    
    upscalers_dir = get_upscalers_dir()
    upscalers_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = upscalers_dir / filename
    
    if output_path.exists():
        logger.info(f"Upscaler already exists: {output_path}")
        return True
    
    logger.info(f"Downloading {name} ({size_mb}MB)...")
    if progress_callback:
        progress_callback(0.0, f"Downloading {name} ({size_mb}MB)...")
    
    try:
        # Download with progress tracking
        temp_path = output_path.with_suffix(".tmp")
        
        def reporthook(count, block_size, total_size):
            if total_size > 0 and progress_callback:
                progress = min(count * block_size / total_size, 1.0)
                progress_callback(progress, f"Downloading {name}: {int(progress * 100)}%")
        
        urllib.request.urlretrieve(url, str(temp_path), reporthook)
        
        # Rename temp file to final name
        temp_path.rename(output_path)
        
        logger.info(f"Downloaded: {output_path}")
        if progress_callback:
            progress_callback(1.0, f"Downloaded {name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        if progress_callback:
            progress_callback(0.0, f"Failed: {e}")
        
        # Clean up temp file if it exists
        temp_path = output_path.with_suffix(".tmp")
        if temp_path.exists():
            temp_path.unlink()
        
        return False


def download_default_upscaler(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """
    Download the default upscaler if no upscalers are installed.
    
    Returns:
        True if an upscaler is available (already installed or downloaded)
    """
    installed = list_installed_upscalers()
    if installed:
        logger.info(f"Upscalers already installed: {installed}")
        return True
    
    logger.info(f"No upscalers found, downloading {DEFAULT_UPSCALER}...")
    return download_upscaler(DEFAULT_UPSCALER, progress_callback)


def download_all_upscalers(
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, bool]:
    """
    Download all available upscalers.
    
    Returns:
        Dict mapping upscaler name to success status
    """
    results = {}
    total = len(AVAILABLE_UPSCALERS)
    
    for i, name in enumerate(AVAILABLE_UPSCALERS):
        def wrapped_callback(progress, message):
            if progress_callback:
                overall_progress = (i + progress) / total
                progress_callback(overall_progress, message)
        
        results[name] = download_upscaler(name, wrapped_callback)
    
    return results


def get_upscaler_info() -> str:
    """Get formatted info about available upscalers."""
    lines = ["**Available Upscalers:**\n"]
    
    installed = list_installed_upscalers()
    
    for name, info in AVAILABLE_UPSCALERS.items():
        status = "✅" if name in installed or info["filename"].replace(".pth", "") in installed else "⬇️"
        lines.append(f"{status} **{name}** ({info['size_mb']}MB)")
        lines.append(f"   {info['description']}")
    
    if not installed:
        lines.append("\n*No upscalers installed. Click Download to get started.*")
    
    return "\n".join(lines)
