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
