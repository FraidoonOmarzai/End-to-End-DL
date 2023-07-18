import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

AUTHOR_USER_NAME = "Fraidoon Omarzai"
SRC_REPO = "PnemobiaDisease"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    description="Using deep learning in medical images",
    long_description=long_description,
    long_description_content="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)