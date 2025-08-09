from setuptools import setup, find_packages

setup(
    name='proto_vla',
    version='0.1.0',
    description='COMPSCI 792 Research Project',
    author='Tony Wang',
    packages=find_packages(include=['proto_vla*']),
    install_requires=[
        "torch==2.7.0",
        "transformers",
        "tqdm",
        "opencv-python",
        "einops",
        "omegaconf",
        "numpy==1.26.0",
    ],
    python_requires=">=3.10",
)

