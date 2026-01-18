import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def merge_configs(base: Dict, override: Dict) -> Dict:
    """Merge two configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def get_config(config_name: str, override_path: str = None) -> Dict:
    """Get configuration by name with optional overrides."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    config_dir = os.path.join(base_dir, "configs")
    
    # Map config names to paths
    config_map = {
        "model-28b": "model/wildkatze-28b.yaml",
        "model-7b": "model/wildkatze-7b-edge.yaml",
        "model-70b": "model/wildkatze-70b-research.yaml",
        "pretrain": "training/pretrain.yaml",
        "finetune": "training/finetune.yaml",
        "production": "deployment/production.yaml",
        "development": "deployment/development.yaml",
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown config: {config_name}")
        
    config_path = os.path.join(config_dir, config_map[config_name])
    config = load_config(config_path)
    
    if override_path:
        override = load_config(override_path)
        config = merge_configs(config, override)
        
    return config
