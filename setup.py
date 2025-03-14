from setuptools import setup, find_packages

setup(
    name="maggie",
    version="0.1.0",
    packages=find_packages(include=["maggie", "maggie.*"]),
    install_requires=[
        # Dependencies remain the same
    ],
    python_requires=">=3.10, <3.11",
)