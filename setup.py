"""Legacy setuptools setup for backward compatibility."""
from setuptools import setup, find_packages

setup(
    name="quaton-d2nn",
    version="0.1.0",
    packages=find_packages(include=["d2nn*"]),
)
