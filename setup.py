import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="route-gym",
    version="0.0.11",
    description="RL environment for solving shortest or longest route problems",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Velythyl/route-gym",
    author="Charlie Gauthier",
    author_email="charlie.gauthier@umontreal.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("img")),
    include_package_data=True,
    install_requires=["networkx", "pyglet", "numpy", "matplotlib", "pymdptoolbox"]
)