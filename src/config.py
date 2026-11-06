"""YAML config loading with `_base_` inheritance and dotted overrides."""
import copy
import os

import yaml


def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path, overrides=None):
    """Load a YAML config. A `_base_: other.yaml` key (relative path) is merged first.
    `overrides` is a list like ["train.epochs=10", "reg.nuclear_emb=1e-3"]."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_cfg = load_config(os.path.join(os.path.dirname(path), base))
        cfg = deep_update(base_cfg, cfg)
    for o in overrides or []:
        k, v = o.split("=", 1)
        set_nested(cfg, k, yaml.safe_load(v))
    return cfg
