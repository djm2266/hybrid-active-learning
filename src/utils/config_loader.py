#!/usr/bin/env python3
"""Configuration utilities"""
import yaml
import os


def load_config(config_path='config/config.yaml'):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_directories(config):
    """Create necessary directories"""
    for dir_path in config['directories'].values():
        os.makedirs(dir_path, exist_ok=True)
