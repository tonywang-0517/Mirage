from setuptools import setup, find_packages

setup(
    name="gr00t",
    version="0.1.0",
    packages=["gr00t"] + [f"gr00t.{p}" for p in find_packages(where=".")],
    package_dir={"gr00t": "."},  # 当前目录就是 protomotions 包
)
