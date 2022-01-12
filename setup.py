import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chefboost",
    version="0.0.17",
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
    install_requires=["pandas>=0.22.0", "numpy>=1.14.0", "tqdm>=4.30.0", "psutil>=5.4.3"]
)
