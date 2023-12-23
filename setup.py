import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setuptools.setup(
    name="chefboost",
    version="0.0.18",
    author="Sefik Ilkin Serengil",
    author_email="serengil@gmail.com",
    description="Lightweight Decision Tree Framework Supporting GBM, Random Forest and Adaboost",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/serengil/chefboost",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
