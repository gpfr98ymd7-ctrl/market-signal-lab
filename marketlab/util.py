"""Small helpers shared across modules."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def clean(o):
    """Make a nested structure JSON-safe: numpy types to Python, NaN/inf to None, dates to strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    if o is pd.NaT or o is pd.NA:
        return None
    return o


def write_json_atomic(path, obj):
    """Write to a temporary file, then rename, so a crash can never leave a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1, allow_nan=False))
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None
