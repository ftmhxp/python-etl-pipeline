#!/usr/bin/env python3
"""
Setup script for COVID-19 ETL Pipeline project.
"""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="covid19-etl-pipeline",
    version="1.0.0",
    description="ETL pipeline for COVID-19 data analysis and prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/covid19-etl-pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "notebook>=7.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "covid-etl=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="covid19 etl pipeline data-analysis",
    python_requires=">=3.8",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/covid19-etl-pipeline/issues",
        "Source": "https://github.com/yourusername/covid19-etl-pipeline",
    },
)
