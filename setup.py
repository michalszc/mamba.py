from setuptools import setup, find_packages

setup(
    name='mambapy',
    version='1.2.0+mps',
    packages=find_packages(),
    package_data={'mambapy.mps': ['*.metal']},
    install_requires=[
        'torch>=2.11.0',
        'einops>=0.8.1',
    ],
    author='Alexandre TL',
    author_email='alexandretl3434@gmail.com',
    description='Mamba in PyTorch with a Metal backend for Mamba-1 on Apple Silicon.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/michalszc/mamba.py',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
