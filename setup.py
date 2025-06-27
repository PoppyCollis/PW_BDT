from setuptools import setup, find_packages

setup(
    name='pw_bdt',
    version='0.1.0',
    description='Bayesian decision tree package',
    author='Your Name',
    packages=find_packages(),  # Automatically finds pw_bdt
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
    ],
    python_requires='>=3.7',
)