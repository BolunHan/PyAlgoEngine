import setuptools
import os
import codecs


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")

setuptools.setup(
    name="PyAlgoEngine",
    version=get_version(os.path.join('AlgoEngine', '__init__.py')),
    author="Bolun.Han",
    author_email="Bolun.Han@outlook.com",
    description="Basic algo engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BolunHan/PyAlgoEngine",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pandas',
        'exchange_calendars',
        'PyQuantKit',
        'PyEventEngine',
    ]
)
