# DNA Bendability Analysis

[![Tests](https://github.com/saadsakib/bendability/actions/workflows/tests.yaml/badge.svg)](https://github.com/saadsakib/bendability/actions/workflows/tests.yaml)

This repository contains various utility tools to analyze DNA Bendability. 

## Setup Repository 

**Use virtual environment** 

```sh 
python3 -m venv venv            # Create venv. Only once.
source venv/bin/activate        # Activate venv. 
pip install -r requirements.txt # Install packages. Only once.
```

**Install Source Code** 

To run source code, you need to install it first. Install source code in virtual env as package in editable mode.  

```sh
cd src/
pip install -e .        # Do this everytime venv is activated
```

## Running Code 

Now, you are ready to run. Invoking functionalities is simple. In `src/main.py`, create proper objects and call it's public functions. As an example, for plotting average intrinsic cyclizability around nucleosome dyads in chromosome V, paste the following code in `main.py`. 

```py
from chromosome import Chromosome
from nucleosome import Nucleosome

if __name__ == '__main__':
    Nucleosome(Chromosome('VL')).plot_c0_vs_dist_from_dyad_spread()    
```

Then, run: `python3 main.py`. 

It should create a plot in `figures/nucleosome` directory.

To initialize an object, invoke a function or understand the expected behavior, take a look at corresponding test files, docstrings and function annotations. 

## Directory structure

The structure of this repository is very flat. The modules can be clustered as following:

## Adding Functionalities 

Add new functionalities in a `Class` or in a module if there's already a relevant `Class` or a module for it. Otherwise, create a new module. 

## Testing 

Tests are useful for for making sure that everything is working fine. Apart from running the whole test suite, you can run a single test module or selected tests.

```sh 
cd tests/
python3 -m pytest                          # Runs whole test suite; ~30 mins
python3 -m pytest util/test_reader.py      # Runs a single test module
python3 -m pytest conformation/test_loops.py -k multichrm 
                                           # Runs tests that contains substring 'multichrm' 
```

Following Test Driven Development(TDD) principle, tests should be written whenever new capabilities are added. 

## Miscellaneous

To initialize `SaminRK/DNABendability` submodule:

```sh
git submodule update --init
```