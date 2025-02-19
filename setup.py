# setup.py
from setuptools import setup, find_packages

setup(
    name="patient_positioning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'opencv-contrib-python>=4.8.0',
        'mediapipe>=0.10.0',
        'numpy>=1.24.3'
    ]
)

