import setuptools

setuptools.setup(
    name="ctspy",
    version="0.1.1",
    author="Bingyin Zhao",
    author_email="bingyiz@betterdata.ai",
    description="Conditional time series data generation",
    packages=setuptools.find_packages(),
    url='https://github.com/betterdataai/time-series-synthetic.git',
    license='MIT',
    install_requires=['einops', 'numpy', 'matplotlib', 'scikit-learn', 'wandb', 'pandas', 'supervised-fcn-2', 'numba', 'x-transformers==1.31.6', 'torch==2.2.1', 'pytorch-lightning==2.3.1', 'seaborn']

)