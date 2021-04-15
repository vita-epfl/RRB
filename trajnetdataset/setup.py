"""setup trajnetdataset"""

from setuptools import setup

# extract version from __init__.py
with open('trajnetdataset/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


setup(
    name='trajnetdataset',
    version=VERSION,
    packages=[
        'trajnetdataset',
    ],
    license='MIT',
    description='RRB code.',
    long_description=open('README.rst').read(),
    author='Mohammadhossein Bahari',
    author_email='mohammadhossein.bahari@epfl.ch',

    install_requires=[
        'pysparkling',
        'scipy',
        'trajnettools',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
        'plot': [
            'matplotlib',
        ]
    },
)
