from setuptools import setup, find_packages

setup(
    name='mirage',
    version='0.1.0',
    description='COMPSCI 792 Research Project',
    author='Tony Wang',
    packages=find_packages(include=['mirage*']),
    python_requires=">=3.10",
)

