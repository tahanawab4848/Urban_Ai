"""
Utility functions for plotting, metric computation, and artifact management.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Official CPCB Palette Mapping
CPCB_COLOR_MAP = {
    "Good": "#00E400",          # Bright Green
    "Satisfactory": "#70A800",  # Light Green / Olive
    "Moderate": "#E6D800",      # Yellow
    "Poor": "#FF7E00",          # Orange
    "Very Poor": "#FF0000",     # Red
    "Severe": "#7E0023",        # Maroon
}

ORDERED_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, filepath: str):
    """Save dictionary to formatted JSON."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def set_plotting_style():
    """Apply consistent styling across all scientific figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.dpi"] = 300
