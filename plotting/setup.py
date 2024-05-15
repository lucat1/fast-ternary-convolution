from setuptools import find_packages
from setuptools import setup

setup(
    name='plot',
    version='0.1.0',
    install_requires=[],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    author='ASL Team 19',
    author_email='noreply@ethz.ch',
    description='Plotting for ASL Project in FS2024'
)
