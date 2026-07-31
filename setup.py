"""Package metadata for the RSF-DETR research implementation."""

import re
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def get_version():
    """Read the public package version."""
    text = (ROOT / "ultralytics/__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', text, re.M)[1]


def parse_requirements(path):
    """Read non-comment dependency lines."""
    return [
        line.split("#", 1)[0].strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="rsf-detr",
    version=get_version(),
    python_requires=">=3.8",
    license="AGPL-3.0",
    description="Official implementation of RSF-DETR for small-object detection.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Anonymous authors",
    packages=find_packages(include=("ultralytics", "ultralytics.*")),
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=parse_requirements(ROOT / "requirements.txt"),
    extras_require={"dev": ["pytest"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="RSF-DETR, object detection, small-object detection, construction-site monitoring",
)
