import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'bimanual_instant_policy'
AUTHOR = 'Srinivas'
AUTHOR_EMAIL = 'you@email.com'
URL = 'https://github.com/sri299792458/bimanual_instant_policy'

LICENSE = 'MIT'
DESCRIPTION = 'Bimanual Instant Policy for dual-arm manipulation'

INSTALL_REQUIRES = [
    'torch>=2.0',
    'torch_geometric',
    'lightning',
    'numpy',
    'scipy',
    'diffusers',
    'wandb',
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(include=['src', 'src.*', 'external', 'external.*']),
    python_requires='>=3.8',
)
