# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "YOM Word2Vec Recommender"
copyright = "2026, Christian Hagemeyer"
author = "Christian Hagemeyer"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_mock_imports = [
    "zenml",
    "polars",
    "gensim",
    "numpy",
    "matplotlib",
    "sklearn",
    "lightgbm",
    "pandas",
    "pyarrow",
    "tqdm",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    'style_nav_header_background': '#2862a3',
}

html_context = {
    'display_github': True,
    'github_user': 'chris026',
    'github_repo': 'YOM-Word2Vec',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

html_static_path = []
