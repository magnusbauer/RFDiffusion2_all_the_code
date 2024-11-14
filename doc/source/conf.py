# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rf_diffusion'
copyright = '2024, Baker Lab'
author = 'Baker Lab'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx_mdinclude',

]

suppress_warnings = [
    'app.add_node',
    'app.add_directive',
    'app.add_role',
    'app.add_generic_role',
    'app.add_source_parser',
    'config.cache',
    'download.not_readable',
    'epub.unknown_project_files',
    'epub.duplicated_toc_entry',
    'i18n.inconsistent_references',
    'index',
    'image.not_readable',
    'ref.term',
    'ref.ref',
    'ref.numref',
    'ref.keyword',
    'ref.option',
    'ref.citation',
    'ref.footnote',
    'ref.doc',
    'ref.python',
    'misc.highlighting_failure',
    'toc.circular',
    'toc.excluded',
    'toc.not_readable',
    'toc.secnum',
    # autodoc,
    # autodoc.import_object,
    # autosummary,
    # intersphinx.external,
]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
