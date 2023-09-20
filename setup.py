from setuptools import setup, find_packages

setup(
    name = 'rela-tensorflow',
    packages = find_packages(exclude = []),
    version = '0.0.1',
    license = 'MIT',
    description = 'ReLA - Sparse Attention with Linear Units - TensorFlow',
    author = 'kevinyecs',
    author_email = '',
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/kevinyecs/rela_tensorflow',
    keywords = [
        'artificial intelligence',
        'deep learning',
        'transformers',
        'attention mechanism',
        'rectified linear attention'
    ],
    install_requires = [
        'tensorflow>=2.13.0',
    ],
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: MIT License',
        'Programming Language :: Python :: 3.11'
    ]
)
