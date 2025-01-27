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
    install_requires=['imbalanced-learn==0.12.3', 'imblearn==0.0', 'jsonschema==4.23.0', 'pycountry==22.3.5', 'scipy==1.14.0', 'transformers==4.40.2', 'tqdm==4.66.5',
                      'joblib==1.4.2', 'pdoc3==0.11.1', 'docstring_parser==0.16', 'psutil==5.9.0', 'rstr==3.2.2', 'pytorch_tabnet==4.1.0',
                      'einops', 'numpy==1.26.4', 'matplotlib', 'scikit-learn==1.4.2', 'wandb', 'pandas==2.2.2', 'supervised-fcn-2', 'numba', 'x-transformers==1.31.6',
                      'torch==2.4.0', 'pytorch-lightning==2.3.1', 'seaborn']

)