from pathlib import Path
import re


project = "DustPy-GPU"
copyright = "2026, Rixin Li and the DustPy authors (Sebastian Stammler and Tilman Birnstiel)"
author = "Rixin Li and the DustPy authors"

pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
release = re.search(r"^version = ['\"]([^'\"]+)", pyproject, re.MULTILINE).group(1)
version = release

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "A_citation.ipynb",
    "B_publications.ipynb",
    "C_contrib_bug_feature.ipynb",
    "D_discussions.ipynb",
    "E_changelog.ipynb",
    "dustpylib.ipynb",
]

nbsphinx_execute = "never"
rst_epilog = f".. |release| replace:: {release}"

html_theme = "sphinx_rtd_theme"
html_title = f"DustPy-GPU {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo-gpu.png"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 4,
    "titles_only": False,
}
html_context = {
    "display_github": True,
    "github_user": "astroboylrx",
    "github_repo": "dustpy-gpu",
    "github_version": "master",
    "conf_py_path": "/docs_source/source/",
}
