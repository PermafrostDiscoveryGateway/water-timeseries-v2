# Documentation Setup Guide

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
cd docs
pip install -r requirements.txt
```

Or install all development dependencies:

```bash
pip install -e ".[dev]"
```

### Building HTML Documentation

Navigate to the docs directory and build:

```bash
cd docs
sphinx-build -W --keep-going -b html . _build/html
```

Or use the provided Makefile:

```bash
cd docs
make html
```

### Viewing Documentation Locally

After building, serve the documentation:

```bash
cd docs
python -m http.server -d _build/html 8000
```

Then open your browser to: `http://localhost:8000`

### Live Rebuild (Auto-rebuild on changes)

For development, use sphinx-autobuild:

```bash
pip install sphinx-autobuild
cd docs
make livehtml
```

This will automatically rebuild the documentation when you change files.

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Documentation homepage
├── getting_started.md   # Installation and quick start guide
├── examples.md          # Usage examples
├── api/                 # API documentation (auto-generated)
│   ├── index.rst
│   ├── dataset.rst      # Dataset classes documentation
│   ├── breakpoint.rst   # Breakpoint detection documentation
│   └── __init__.rst     # Main module documentation
├── _build/              # Build output (excluded from git)
├── Makefile             # Build commands
└── requirements.txt     # Documentation dependencies
```

## Adding Documentation

### Adding New Modules

To document a new module, create an `.rst` file in `docs/api/`:

```rst
my_module Module
=================

.. automodule:: water_timeseries.my_module
   :members:
   :undoc-members:
   :show-inheritance:
```

Then add it to `docs/api/index.rst`.

### Writing Docstrings

Use Google-style docstrings in your code:

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description.
    
    Longer description explaining what the function does,
    its behavior, and important notes.
    
    Args:
        param1 (str): Description of param1.
        param2 (int): Description of param2.
    
    Returns:
        bool: Description of return value.
    
    Example:
        >>> result = my_function("test", 42)
        >>> print(result)
        True
    
    Raises:
        ValueError: When invalid parameters are provided.
    """
    pass
```

### Writing Examples

Add example notebooks to the docs by:

1. Creating a Jupyter notebook in `notebooks/`
2. Reference it in `docs/examples.md` using MyST link syntax

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages on every push to `main` via the GitHub Actions workflow in `.github/workflows/docs.yml`.

The workflow:
1. Triggers on push to `main` branch
2. Installs dependencies and builds HTML
3. Deploys to GitHub Pages
4. Makes documentation available at: `https://PermafrostDiscoveryGateway.github.io/water-timeseries-v2/`

## Configuration

Documentation configuration is in `docs/conf.py`:

- **Theme**: Uses the modern Furo theme
- **Extensions**: 
  - `sphinx.ext.autodoc` - Auto-generate API docs from docstrings
  - `sphinx.ext.napoleon` - Support for Google/NumPy style docstrings
  - `myst_parser` - Markdown support
  - `sphinx.ext.viewcode` - Source code links in API docs

## Troubleshooting

### Build failures

If the build fails:

1. Check for warnings in output (use `-W` flag to treat warnings as errors)
2. Check docstring formatting
3. Verify imports work: `python -c "import water_timeseries"`

### Missing API documentation

Ensure:
1. Module has docstrings
2. Module is imported in `water_timeseries/__init__.py`
3. API .rst file exists and is referenced in `api/index.rst`
