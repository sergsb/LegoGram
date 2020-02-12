import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="legogram", # Replace with your own username
    version="0.0.1",
    author="Sergey Sosnin",
    author_email="sergey.sosnin@skoltech.ru",
    description="LegoGram: molecular grammars for de-no vo generation of chemical compounds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sergsb/legogram",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL3 License",
        "Operating System :: OS Independent",
    ],
    package_data={'legogram': ['data/*','models/*']},
    python_requires='>=3.6'
)