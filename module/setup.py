from setuptools import setup
from codecs import open

# Get information from separate files (README, VERSION)
def readfile(filename):
    with open(filename,  encoding="utf-8") as f:
        return f.read()
    
setup(
    name="ztpt",
    version='0.1.0', # the VERSION file is shared with the documentation
    description="Training and fine-tuning transformer-based neural network to perform on SQuAD dataset",
    long_description=readfile("README.md"), # get the long description from the README
    url="https://github.com/zatoboj/zatoboj",
    install_requires=['transformers==3.5.0','pytorch-lightning','torch==1.7.0'],
    author="Efim Abrikosov, Eduard Duryev, Matvey Kolbasov, Leonid Monin, Petr Pushkar, Alexey Savchik, Anton Osinenko",
    author_email="edwardduriev@gmail.com",
    license="MIT",
    classifiers=[
      "Development Status :: 2 - Pre-Alpha",
      "Intended Audience :: Science/Research",
      "Topic :: Software Development :: Build Tools",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
      "License :: OSI Approved :: MIT License",
      "Programming Language :: Python :: 3.6",
    ], # classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    keywords = "transformer squad nli nlp machine learning",
    packages = ["ztpt"],
)