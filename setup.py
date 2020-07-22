from typing import Dict

from setuptools import find_packages, setup

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import cerbero.
VERSION: Dict[str, str] = {}
with open("cerbero/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# Use README.md as the long_description for the package
with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="cerbero",
    version=VERSION["VERSION"],
    url="",
    description="Multi-task learning made easy",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "munkres>=1.0.6",
        "numpy>=1.16.0,<2.0.0",
        "scipy>=1.2.0,<2.0.0",
        "pandas>=0.25.0,<1.1.0",
        "tqdm>=4.33.0,<5.0.0",
        "scikit-learn>=0.20.2,<0.22.0",
        "torch>=1.2.0,<2.0.0",
        "tensorboard>=1.14.0,<2.0.0",
        "networkx>=2.2,<2.4",
    ],
    python_requires=">=3.6",
    keywords="machine-learning multi-task-learning",
)
