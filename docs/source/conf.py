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

html_theme = 'furo'
html_theme_options = {
    'source_repository': 'https://github.com/BolunHan/PyAlgoEngine/',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
}
html_static_path = ['_static']

suppress_warnings = ['import']


def setup(app):
    from sphinx.builders.html import StandaloneHTMLBuilder
    StandaloneHTMLBuilder.supported_image_types = [
        'image/svg+xml',
        'image/png',
        'image/jpeg',
    ]
