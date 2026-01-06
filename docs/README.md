# tritonBLAS Documentation

This directory contains the documentation for tritonBLAS.

## Building the Documentation

### Prerequisites

- Python 3.10+
- Sphinx and required extensions (see `sphinx/requirements.txt`)

### Quick Build

From the `docs/` directory:

```bash
./build_docs.sh
```

This script will:
1. Create a virtual environment (if needed)
2. Install dependencies
3. Build the documentation
4. Output to `_build/html/`

### Manual Build

```bash
# Install dependencies
pip install -r sphinx/requirements.txt

# Build documentation
python3 -m sphinx -b html -d _build/doctrees -D language=en . _build/html
```

### Viewing the Documentation

After building, open `_build/html/index.html` in your browser, or serve it locally:

```bash
python3 -m http.server -d _build/html/
```

Then visit http://localhost:8000

### Auto-rebuild on Changes

For development, use sphinx-autobuild:

```bash
python3 -m sphinx_autobuild -b html -d _build/doctrees -D language=en . _build/html
```

This will automatically rebuild when files change and serve at http://localhost:8000

## Documentation Structure

```
docs/
├── index.md                    # Main landing page
├── conf.py                     # Sphinx configuration
├── build_docs.sh              # Build script
├── .nojekyll                  # GitHub Pages configuration
├── getting-started/           # Getting started guides
│   ├── installation.md
│   ├── quickstart.md
│   └── examples.md
├── conceptual/                # Conceptual documentation
│   ├── analytical-model.md
│   ├── architecture.md
│   └── performance.md
├── reference/                 # API reference
│   ├── api.md
│   ├── configuration.md
│   └── advanced.md
├── sphinx/                    # Sphinx configuration
│   ├── _toc.yml              # Table of contents
│   └── requirements.txt      # Python dependencies
└── _static/                   # Static assets (CSS, images)
```

## Contributing to Documentation

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add it to `sphinx/_toc.yml`
3. Rebuild the documentation

### Writing Style

- Use clear, concise language
- Include code examples where appropriate
- Follow the existing structure and formatting
- Use proper Markdown syntax

### Code Examples

All code examples should be:
- Runnable (when possible)
- Well-commented
- Following best practices

## Publishing

The documentation is configured for:
- **ReadTheDocs**: Via `.readthedocs.yaml` in the repository root
- **GitHub Pages**: Via `.nojekyll` file

### ReadTheDocs

The documentation will automatically build on ReadTheDocs when pushed to the repository.

### GitHub Pages

To publish to GitHub Pages:

1. Build the documentation locally
2. Copy `_build/html/` contents to your GitHub Pages branch
3. Ensure `.nojekyll` is present in the root

## Troubleshooting

### Build Errors

If you encounter build errors:

1. Check that all dependencies are installed
2. Verify Python version (3.10+)
3. Check for syntax errors in `.md` files
4. Review `conf.py` for configuration issues

### Missing Pages

If pages don't appear in the navigation:

1. Verify the page is listed in `sphinx/_toc.yml`
2. Check file paths are correct
3. Rebuild the documentation

### Styling Issues

If styling looks incorrect:

1. Check that `_static/` directory exists
2. Verify `html_static_path` in `conf.py`
3. Clear the build directory and rebuild

## Support

For documentation issues:
- Open an issue on GitHub
- Contact the development team
- Check the Sphinx documentation: https://www.sphinx-doc.org/
