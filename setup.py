from setuptools import setup, find_packages

setup(
    name='bayesian-optimization',
    version='1.0.2',
    url='https://github.com/raff7/Bayesian_Optimization',
    packages=find_packages(),
    author='Fernando M. F. Nogueira, Raffaele Piccini',
    author_email="raffaele.piccini@mailonline.co.uk",
    description='Extended Bayesian Optimization package',
    long_description='A Python implementation of global optimization with gaussian processes.',
    download_url='https://github.com/raff7/Bayesian_Optimization.git',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "scikit-learn >= 0.18.0",
        "sobol_seq"
    ],
)
