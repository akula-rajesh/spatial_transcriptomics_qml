"""
Setup script for spatial transcriptomics ML pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
 readme_path = Path("README.md")
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Spatial Transcriptomics Machine Learning Pipeline"

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spatial-transcriptomics-pipeline",
    version="1.0.0",
    author="Spatial Transcriptomics Research Team",
    author_email="research@spatial-transcriptomics.org",
    description="Machine learning pipeline for spatial transcriptomics analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spatial-transcriptomics/research-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "quantum": [
            "pennylane>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spatial-train=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "spatial transcriptomics",
        "machine learning",
        "deep learning",
        "computational biology",
        "bioinformatics",
        "gene expression",
        "histology",
        "computer vision",
    ],
    project_urls={
        "Bug Reports": "https://github.com/spatial-transcriptomics/research-pipeline/issues",
        "Source": "https://github.com/spatial-transcriptomics/research-pipeline",
        "Documentation": "https://spatial-transcriptomics.github.io/research-pipeline/",
    },
)
