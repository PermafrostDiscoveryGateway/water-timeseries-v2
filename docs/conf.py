# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "water-timeseries"
copyright = "2026, Ingmar Nitze"
author = "Ingmar Nitze"

# Version
release = "0.1.0"
version = "0.1"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Napoleon configuration (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# MyST configuration for Markdown support
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

# HTML theme
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/PermafrostDiscoveryGateway/water-timeseries-v2",
    "source_branch": "main",
    "source_directory": "docs",
}

# Source file suffixes
source_suffix = {
    ".rst": None,
    ".md": "myst-nb",
}

# Exclude patterns
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
