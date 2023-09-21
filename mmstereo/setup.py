from distutils.core import setup

setup(
    name="mmstereo",
    version="0.0.0",
    author="Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, Max Bajracharya",
    author_email="matias@robots.ox.ac.uk",
    packages=["mmstereo"],
    package_dir={"": "."},
    python_requires=">=3.8",
    description="mmstereo package",
)
