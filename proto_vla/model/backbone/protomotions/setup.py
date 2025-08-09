from setuptools import setup, find_packages

setup(
    name="protomotions",
    version="0.1.0",
    packages=["protomotions"] + [f"protomotions.{p}" for p in find_packages(where=".")],
    package_dir={"protomotions": "."},  # 当前目录就是 protomotions 包
)

