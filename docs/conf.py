# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../include"))

# -- Project information -----------------------------------------------------

project = "tritonBLAS"
copyright = "2025, Advanced Micro Devices, Inc."
author = "AMD Research and Advanced Development Team"
# Display "latest" in the docs header instead of a fixed version
release = "latest"
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "rocm_docs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "rocm_docs_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Customize the HTML title shown in the top-left/header
html_title = "tritonBLAS Documentation"

# -- Extension configuration -------------------------------------------------

# Autodoc configuration for generating docs from docstrings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "inherited-members": True,
}

# Show type hints in documentation
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Render objects without full module path
add_module_names = False

# Mock heavy/runtime-only dependencies when building docs
autodoc_mock_imports = [
    "torch",
    "numpy",
    "origami",
]

# Docstring-preserving decorator mock
class PreserveDocstringMock:
    """Mock decorator that preserves docstrings and function attributes.
    
    Handles all decorator patterns:
    - @triton.jit (direct decorator)
    - @triton.jit() (decorator factory with no args)
    - @triton.jit(...) (decorator factory with args)
    - @triton.heuristics({...}) (decorator with dict config)
    """

    def __call__(self, func_or_config=None, **kwargs):
        # If called with a dict/config (like @triton.heuristics({...})),
        # return self to act as the decorator
        if isinstance(func_or_config, dict):
            return self
        # Handle @triton.jit() with parentheses (decorator factory pattern)
        if func_or_config is None:
            return self
        # Handle @triton.jit without parentheses (direct decorator pattern)
        # func_or_config is the actual function, return it unchanged
        return func_or_config


# Create clean mock types for triton.language that display nicely in docs
class TritonTypeMock:
    """Mock for Triton types that displays a clean name in documentation."""
    
    def __init__(self, name):
        self._name = name
    
    def __repr__(self):
        return self._name
    
    def __str__(self):
        return self._name
    
    def __getitem__(self, item):
        # Handle generic types like tl.tensor[...]
        return TritonTypeMock(f"{self._name}[{item}]")
    
    def __call__(self, *args, **kwargs):
        # Some types might be called as constructors
        return self


class TritonLanguageMock:
    """Mock for triton.language module with clean type representations."""
    
    # Common Triton types used in stages
    tensor = TritonTypeMock("tl.tensor")
    constexpr = TritonTypeMock("tl.constexpr")
    pointer_type = TritonTypeMock("tl.pointer_type")
    block_type = TritonTypeMock("tl.block_type")
    int32 = TritonTypeMock("tl.int32")
    int64 = TritonTypeMock("tl.int64")
    float16 = TritonTypeMock("tl.float16")
    float32 = TritonTypeMock("tl.float32")
    bfloat16 = TritonTypeMock("tl.bfloat16")
    
    # Common functions - return MagicMock for flexibility
    load = MagicMock(return_value=TritonTypeMock("tl.tensor"))
    store = MagicMock(return_value=None)
    dot = MagicMock(return_value=TritonTypeMock("tl.tensor"))
    zeros = MagicMock(return_value=TritonTypeMock("tl.tensor"))
    arange = MagicMock(return_value=TritonTypeMock("tl.tensor"))
    program_id = MagicMock(return_value=TritonTypeMock("tl.tensor"))
    num_programs = MagicMock(return_value=TritonTypeMock("tl.tensor"))
    
    def __getattr__(self, name):
        # For any other attributes, return a mock that looks clean
        return TritonTypeMock(f"tl.{name}")


triton_language_mock = TritonLanguageMock()
sys.modules["triton.language"] = triton_language_mock

# Mock triton.language.core with _aggregate decorator support
class TritonLanguageCoreMock:
    @staticmethod
    def _aggregate(cls):
        """Preserve the class when @aggregate decorator is used."""
        return cls
    
    def __getattr__(self, name):
        return TritonTypeMock(f"tl.core.{name}")


sys.modules["triton.language.core"] = TritonLanguageCoreMock()


# Mock triton modules with docstring-preserving jit decorator
class TritonMock:
    jit = PreserveDocstringMock()
    constexpr_function = PreserveDocstringMock()
    language = triton_language_mock
    autotune = PreserveDocstringMock()
    heuristics = PreserveDocstringMock()
    Config = MagicMock()
    cdiv = MagicMock(return_value=1)


sys.modules["triton"] = TritonMock()

# Napoleon settings for Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_warnings = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# ROCm docs handles most configuration automatically
external_projects_current_project = "tritonBLAS"

# Table of contents
external_toc_path = "./sphinx/_toc.yml"

# Theme options for AMD ROCm theme
html_theme_options = {
    "flavor": "instinct",
    "link_main_doc": True,
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_hide = False
copybutton_remove_prompts = True

# Force copy buttons to be generated
html_context = {
    "copybutton_prompt_text": copybutton_prompt_text,
    "copybutton_prompt_is_regexp": copybutton_prompt_is_regexp,
    "copybutton_line_continuation_character": copybutton_line_continuation_character,
    "copybutton_hide": copybutton_hide,
    "copybutton_remove_prompts": copybutton_remove_prompts,
}
