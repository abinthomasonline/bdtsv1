import setuptools

setuptools.setup(
    name="bdtsv1",
    version="0.1.1",
    author="Bingyin Zhao",
    author_email="bingyiz@betterdata.ai",
    description="Conditional time series data generation with relational components",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url='https://github.com/bxz9200/bdtsv1',
    license='MIT',
    install_requires=[
        'einops',
        'numpy>=1.26.4',
        'matplotlib',
        'scikit-learn>=1.4.2',
        'wandb',
        'pandas>=2.2.2',
        'supervised-fcn-2',
        'numba',
        'x-transformers==1.31.6',
        'torch>=2.2.1',
        'pytorch-lightning>=2.3.1',
        'seaborn',
        'scipy>=1.14.0',
        # 'datapip'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    include_package_data=True,
    package_data={
        'tsv1': [
            'configs/*.json',
            'configs/*.yaml',
            'configs/data/*.json',
            'configs/model/*.json'
        ]
    }
)