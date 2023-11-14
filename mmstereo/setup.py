from distutils.core import setup
from setuptools import find_packages

setup(
    name="mmstereo",
    version="0.0.1",
    author="Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, Max Bajracharya",
    author_email="matias@robots.ox.ac.uk",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.8",
    description="mmstereo package",
)
