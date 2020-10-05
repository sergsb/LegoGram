import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.9.1"
    
setuptools.setup(
    name="legogram",
    version=__version__,
    author="I.Khokhlov, S.Sosnin",
    author_email="hohlovivan@gmail.com, sergey.sosnin@skoltech.ru",
    description="Molecular Grammar, simple as LEGO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://no.page.yet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'python-igraph',
          'tqdm',
          'molvs',
          'joblib',
          'pydot',
          'pygraphviz',
          'enums'        
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
