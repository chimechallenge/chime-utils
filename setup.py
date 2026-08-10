from pathlib import Path

from setuptools import find_packages, setup

requirements = []

try:
    import torch  # noqa: F401
except ImportError:
    requirements.append("torch")


def read_requirements():
    requirements_file = Path(__file__).resolve().parent / "requirements.txt"

    lines = requirements_file.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="chime_utils",
    py_modules=["chime_utils"],
    version="0.0.1",
    description="Common script for CHiME-8 Task 1 & 2.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    author="Samuele Cornell",
    url="https://www.chimechallenge.org/",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=requirements + read_requirements(),
    entry_points={
        "console_scripts": [
            "chime-utils=chime_utils.bin.base:cli",
        ]
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "scipy", "black", "flake8", "isort"]},
)
