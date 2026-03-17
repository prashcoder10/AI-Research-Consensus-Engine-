import os
import yaml
from typing import Any, Dict, Optional
from functools import lru_cache

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

# ==========================================
# Core Loader (Singleton via caching)
# ==========================================

@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation and environment overrides.

    Args:
        config_path (str, optional): Path to config file

    Returns:
        dict: Loaded configuration
    """
    config_path = config_path or os.getenv("CONFIG_PATH", "config/settings.yaml")

    if not os.path.exists(config_path):
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML format: {str(e)}")

    if not isinstance(config, dict):
        raise ConfigError("Config file must contain a valid YAML dictionary")

    # Apply environment overrides
    config = _apply_env_overrides(config)

    # Validate required fields
    _validate_config(config)

    return config

# ==========================================
# Environment Overrides
# ==========================================

def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override config values using environment variables.

    Format:
        ENV VAR: APP__DEBUG=true
        Maps to: config['app']['debug'] = True
    """
    for key, value in os.environ.items():
        if "__" not in key:
            continue

        keys = key.lower().split("__")
        _set_nested_value(config, keys, _parse_env_value(value))

    return config

def _set_nested_value(config: Dict[str, Any], keys: list, value: Any):
    """
    Set value in nested dictionary using list of keys.
    """
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]

    d[keys[-1]] = value

def _parse_env_value(value: str) -> Any:
    """
    Convert environment variable string to proper type.
    """
    value = value.strip()

    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

# ==========================================
# Validation
# ==========================================

def _validate_config(config: Dict[str, Any]):
    """
    Validate required configuration fields.
    Raises ConfigError if invalid.
    """
    required_fields = [
        ("app", dict),
        ("embedding", dict),
        ("vector_store", dict),
        ("llm", dict),
    ]

    for field, expected_type in required_fields:
        if field not in config:
            raise ConfigError(f"Missing required config section: '{field}'")

        if not isinstance(config[field], expected_type):
            raise ConfigError(
                f"Invalid type for '{field}': expected {expected_type}, got {type(config[field])}"
            )

    # Specific validations
    if "model_name" not in config["embedding"]:
        raise ConfigError("Missing 'embedding.model_name'")

    if config["embedding"].get("batch_size", 0) <= 0:
        raise ConfigError("embedding.batch_size must be > 0")

    if config.get("retrieval", {}).get("top_k", 0) <= 0:
        raise ConfigError("retrieval.top_k must be > 0")

    if config.get("limits", {}).get("max_input_sources", 0) <= 0:
        raise ConfigError("limits.max_input_sources must be > 0")

# ==========================================
# Helper Access Functions
# ==========================================

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Access config using dot notation.

    Example:
        get_config_value("embedding.model_name")
    """
    config = load_config()
    keys = key.split(".")

    value = config
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            return default
        value = value[k]

    return value

def reload_config():
    """
    Clear cache and reload config (useful in dev/testing).
    """
    load_config.cache_clear()