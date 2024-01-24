from setuptools import setup, find_packages
from pathlib import Path

readme = Path('README.md').read_text(encoding='utf-8')

setup(
    name='tdw_image_dataset',
    version="1.1.0",
    description='Generate synthetic image datasets with TDW',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/alters-mit/tdw_image_dataset',
    author_email='alters@mit.edu',
    author='Seth Alter',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    packages=find_packages(),
    include_package_data=True,
    keywords='unity simulation ml machine-learning',
    install_requires=['tqdm', 'numpy', 'tdw'],
)
