"""
Filename utilities for multi-generation scenarios.
Provides functions to generate unique filenames that include main parameters, seed, and unique identifiers.
"""

import os
import uuid
import hashlib
import re
from datetime import datetime
from typing import Dict, Any, Optional


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """
    Sanitize text for use in filenames by removing/replacing problematic characters.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum length for the sanitized text
        
    Returns:
        Sanitized text suitable for filenames
    """
    if not text:
        return "empty"
    
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', text)
    sanitized = re.sub(r'\s+', '_', sanitized)  # Replace whitespace with underscores
    sanitized = re.sub(r'_+', '_', sanitized)   # Collapse multiple underscores
    sanitized = sanitized.strip('._')           # Remove leading/trailing dots and underscores
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized or "unnamed"


def condense_prompt(prompt: str, max_length: int = 30) -> str:
    """
    Condense a prompt for filename usage by keeping important words.
    
    Args:
        prompt: Input prompt text
        max_length: Maximum length for condensed prompt
        
    Returns:
        Condensed prompt suitable for filenames
    """
    if not prompt:
        return "no_prompt"
    
    # Remove common stop words and keep meaningful words
    stop_words = {'a', 'an', 'and', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = prompt.lower().split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    
    # Take first few meaningful words
    condensed = '_'.join(filtered_words[:5])
    condensed = sanitize_filename(condensed, max_length)
    
    return condensed


def generate_unique_filename(
    base_params: Dict[str, Any],
    seed: int,
    file_extension: str = "wav",
    output_dir: str = ".",
    naming_style: str = "detailed",
    include_timestamp: bool = True,
    include_uuid: bool = True
) -> str:
    """
    Generate a unique filename that includes main parameters, seed, and unique identifier.
    
    Args:
        base_params: Dictionary of key generation parameters
        seed: Random seed used for generation
        file_extension: File extension (without dot)
        output_dir: Output directory path
        naming_style: Style of filename generation ("detailed", "compact", "minimal")
        include_timestamp: Whether to include timestamp in filename
        include_uuid: Whether to include UUID for uniqueness
        
    Returns:
        Full path to unique filename
        
    Example:
        base_params = {
            "prompt": "piano music with jazz elements",
            "steps": 100,
            "cfg_scale": 7.5,
            "model_name": "stable-audio-v1",
            "sampler": "dpmpp-3m-sde"
        }
        # Returns: ./piano_music_jazz_elements_s100_cfg7.5_seed12345_20240101_123456_abc12345.wav
    """
    
    # Extract and sanitize key parameters
    prompt = base_params.get("prompt", "")
    steps = base_params.get("steps", 50)
    cfg_scale = base_params.get("cfg_scale", 7.0)
    model_name = base_params.get("model_name", "model")
    sampler = base_params.get("sampler", "default")
    
    # Generate filename components based on style
    components = []
    
    if naming_style == "detailed":
        # Include all major parameters
        if prompt:
            components.append(condense_prompt(prompt, 25))
        components.append(f"s{steps}")
        components.append(f"cfg{cfg_scale}")
        components.append(f"seed{seed}")
        if model_name:
            model_short = sanitize_filename(model_name.split('/')[-1], 15)  # Handle HF model names
            components.append(f"m{model_short}")
        if sampler and sampler != "default":
            sampler_short = sanitize_filename(sampler, 10)
            components.append(f"smp{sampler_short}")
            
    elif naming_style == "compact":
        # Include essential parameters only
        if prompt:
            components.append(condense_prompt(prompt, 20))
        components.append(f"s{steps}")
        components.append(f"cfg{cfg_scale}")
        components.append(f"seed{seed}")
        
    elif naming_style == "minimal":
        # Just prompt and seed
        if prompt:
            components.append(condense_prompt(prompt, 15))
        components.append(f"seed{seed}")
    
    # Add timestamp if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        components.append(timestamp)
    
    # Add unique identifier if requested
    if include_uuid:
        # Generate short UUID based on all parameters for reproducibility
        param_string = f"{prompt}_{steps}_{cfg_scale}_{seed}_{model_name}_{sampler}"
        param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
        components.append(param_hash)
    
    # Combine all components
    filename_base = "_".join(components)
    
    # Ensure filename isn't too long (most filesystems have 255 char limit)
    max_base_length = 200  # Leave room for extension and path
    if len(filename_base) > max_base_length:
        filename_base = filename_base[:max_base_length]
    
    # Create full filename with extension
    filename = f"{filename_base}.{file_extension}"
    
    # Handle potential conflicts by adding incrementing number
    full_path = os.path.join(output_dir, filename)
    counter = 1
    original_base = filename_base
    
    while os.path.exists(full_path):
        filename_base = f"{original_base}_{counter:03d}"
        filename = f"{filename_base}.{file_extension}"
        full_path = os.path.join(output_dir, filename)
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 999:
            # Add random UUID if we somehow get 1000 conflicts
            random_suffix = str(uuid.uuid4())[:8]
            filename_base = f"{original_base}_{random_suffix}"
            filename = f"{filename_base}.{file_extension}"
            full_path = os.path.join(output_dir, filename)
            break
    
    return full_path


def generate_batch_filenames(
    base_params_list: list,
    seeds: list,
    file_extension: str = "wav",
    output_dir: str = ".",
    naming_style: str = "detailed"
) -> list:
    """
    Generate unique filenames for a batch of generations.
    
    Args:
        base_params_list: List of parameter dictionaries for each generation
        seeds: List of seeds corresponding to each generation
        file_extension: File extension (without dot)
        output_dir: Output directory path
        naming_style: Style of filename generation
        
    Returns:
        List of unique file paths
    """
    if len(base_params_list) != len(seeds):
        raise ValueError("base_params_list and seeds must have the same length")
    
    filenames = []
    for params, seed in zip(base_params_list, seeds):
        filename = generate_unique_filename(
            base_params=params,
            seed=seed,
            file_extension=file_extension,
            output_dir=output_dir,
            naming_style=naming_style,
            include_timestamp=True,
            include_uuid=True
        )
        filenames.append(filename)
    
    return filenames


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract generation parameters from a filename created by generate_unique_filename.
    
    Args:
        filename: Filename to parse
        
    Returns:
        Dictionary of extracted parameters
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    params = {}
    
    # Extract seed
    seed_match = re.search(r'seed(\d+)', basename)
    if seed_match:
        params['seed'] = int(seed_match.group(1))
    
    # Extract steps
    steps_match = re.search(r's(\d+)', basename)
    if steps_match:
        params['steps'] = int(steps_match.group(1))
    
    # Extract CFG scale
    cfg_match = re.search(r'cfg([\d.]+)', basename)
    if cfg_match:
        params['cfg_scale'] = float(cfg_match.group(1))
    
    # Extract model name
    model_match = re.search(r'm([^_]+)', basename)
    if model_match:
        params['model_name'] = model_match.group(1)
    
    # Extract sampler
    sampler_match = re.search(r'smp([^_]+)', basename)
    if sampler_match:
        params['sampler'] = sampler_match.group(1)
    
    return params