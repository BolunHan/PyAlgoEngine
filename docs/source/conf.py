import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

from algo_engine import __version__

project = 'PyAlgoEngine'
author = 'Han Bolun'
release = __version__
version = '.'.join(__version__.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

autosummary_generate = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}
html_static_path = ['_static']
html_context = {
    'display_github': True,
    'github_user': 'BolunHan',
    'github_repo': 'PyAlgoEngine',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

suppress_warnings = ['import']


def setup(app):
    from sphinx.builders.html import StandaloneHTMLBuilder
    StandaloneHTMLBuilder.supported_image_types = [
        'image/svg+xml',
        'image/png',
        'image/jpeg',
    ]
