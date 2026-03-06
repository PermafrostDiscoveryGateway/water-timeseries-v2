# Documentation Setup Guide

This project uses **mkdocs** with the Material theme for documentation.

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Or install all development dependencies:

```bash
pip install -e ".[dev]"
```

### Building HTML Documentation

From the project root, build the documentation:

```bash
mkdocs build
```

This generates the static site in the `site/` directory.

### Viewing Documentation Locally

Serve the documentation locally with live reload during development:

```bash
mkdocs serve
```

Then open your browser to: `http://localhost:8000`

The documentation will automatically rebuild when you change files.

## Documentation Structure

```
docs/
├── index.md             # Documentation homepage
├── getting_started.md   # Installation and quick start guide
├── examples.md          # Usage examples
├── api/                 # API documentation (auto-generated from code)
│   ├── index.md
│   ├── dataset.md       # Dataset classes documentation
│   ├── breakpoint.md    # Breakpoint detection documentation
│   └── utils.md         # Utils documentation
└── requirements.txt     # Documentation dependencies

site/                    # Build output (excluded from git)
mkdocs.yml              # mkdocs configuration (project root)
```

## Adding Documentation

### Adding New Modules

To document a new module:

1. Create a new file `docs/api/my_module.md`

```markdown
# My Module

::: water_timeseries.my_module
    options:
      show_source: true
      docstring_style: google
```

2. Add it to `mkdocs.yml` in the `nav` section under "API Reference".

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
2. Reference it in `docs/examples.md` with markdown links

## Automatic Deployment

Documentation is automatically built and deployed to GitHub Pages on every push to `main` via the GitHub Actions workflow in `.github/workflows/docs.yml`.

The workflow:

1. Triggers on push to `main` branch
2. Installs dependencies and builds with `mkdocs build`
3. Deploys to GitHub Pages
4. Makes documentation available at: `https://PermafrostDiscoveryGateway.github.io/water-timeseries-v2/`

## Configuration

Documentation configuration is in `mkdocs.yml`:

- **Theme**: Material theme with custom color scheme
- **Plugins**:
  - `search` - Full-text search functionality
  - `mkdocstrings[python]` - Auto-generate API docs from docstrings
- **Extensions**:
  - Python docstring parsing using Google style
  - Math support with LaTeX
  - Code highlighting and other markdown extensions

## Troubleshooting

### Build failures

If the build fails:

1. Check for import errors: `python -c "import water_timeseries"`
2. Verify docstring formatting (Google style)
3. Check that modules are properly defined in source

### Missing API documentation

Ensure:

1. Module has proper docstrings
2. The module file exists in `src/water_timeseries/`
3. The corresponding `.md` file exists in `docs/api/`
4. The path is registered in `mkdocs.yml` under the `nav` section
