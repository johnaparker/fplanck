import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

NAME = 'fplanck'
DESCRIPTION = "Solve the Fokker-Planck equation in N dimensions"
URL = 'https://github.com/johnaparker/fplanck'
EMAIL = 'japarker@uchicago.edu'
AUTHOR = 'John Parker'
KEYWORDS = 'fokker planck smoluchowski physics probability stochastic'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.2.2'
LICENSE = 'MIT'

REQUIRED = [
    'numpy', 
    'scipy',
]

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
    packages=find_packages(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=REQUIRED,
    python_requires=REQUIRES_PYTHON,
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
    ],
)
