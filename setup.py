from setuptools import setup, find_packages

setup(
    name="femto",
    description="Python suite for the design of femtosecond-written circuit.",
    version="1.1",
    packages=find_packages(
        include=["femto", "compiler", "compiler.*", "objects", "objects.*"]
    ),
    install_requires=[
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "plotly",
        "pyclipper",
        "scipy",
    ],
)
