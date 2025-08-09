from setuptools import setup, find_packages

setup(
    name='protoVLA',
    version='0.1.0',
    description='COMPSCI 792 Research Project',
    author='Tony Wang',
    packages=find_packages(include=['protoVLA*', 'protoMotions*']),
    install_requires=[
        "torch>=2.0",
        "transformers",
        "tqdm",
        "opencv-python",
        "einops",
        "omegaconf",
        "numpy",
    ],
    python_requires=">=3.10",
)
