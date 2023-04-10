from setuptools import setup, find_packages
"""
Run this setup.py file to install requirements for this project.
"""
setup(
    name='Pendant-Drop-Tensiometry-Regressor',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'imageio',
        'scipy',
        'numpy',
        'matplotlib',
        'sklearn',
        'xgboost',
        'pandas',
        'circle-fit',
        'pickle',
        'git+https://github.com/DmitriLyalikov/pdt-canny-edge-detector.git'
    ],
    description="Pendant Drop Tensiometry-Regressor",
    author_email='Dlyalikov01@manhattan.edu',
    author='Dmitri Lyalikov'
)