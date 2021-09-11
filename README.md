# DNA Bendability Analysis

[![Tests](https://github.com/saadsakib/bendability/actions/workflows/tests.yaml/badge.svg)](https://github.com/saadsakib/bendability/actions/workflows/tests.yaml)

This repository contains various utility tools to analyze DNA Bendability. 

## Setup Repository 
 
1. First initialize `SaminRK/DNABendability` submodule.  
```sh
git submodule update --init
```

2. Now, Install necessary packages. 
```sh 
pip install -r requirements.txt
```

3. Run `pytest` to check all functions are running ok. All tests should pass. First time running this might take about 30 minutes to 1 hour. From next on, test suite should complete faster. 

```sh
python3 -m pytest
```

## Running Code 

Now, you are ready to run. Invoking functionalities is simple. In `main.py`, create proper objects and call it's public functions. As an example, for plotting average intrinsic cyclizability around nucleosome dyads in chromosome V, paste the following code in `main.py`. 

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

**Feature Extraction and ML Model Training**
- `bq.py`
- `analysis.ipynb`
- `correlation.py`
- `data_organizer.py`
- `feat_selector.py`
- `helsep.py`
- `libstat.py`
- `model.py`
- `occurence.py`
- `shape.py`

**Chromosome, Nucleosome, Hi-C Analysis and Prediction by CNN Model**
- `chromosome.py`
- `genes.py`
- `loops.py`
- `nucleosome.py`
- `prediction.py`
- `tad.py`
- `hic/`
- `meuseum_mod/`
- `dna_shape_model/`

**Utility Modules**
- `constants.py`
- `custom_types.py`
- `main.py`
- `reader.py`
- `util.py`

**Testing**
- `conftest.py`
- `pytest.ini`
- `test_*.py`

## Adding Functionalities 

Add new functionalities in a `Class` or in a module if there's already a relevant `Class` or a module for it. Otherwise, create a new module. 

## Testing 

Tests are useful for for making sure that everything is working fine. Apart from running the whole test suite, you can run a single test module or selected tests.

```sh 
python3 -m pytest test_reader.py             # Runs a single test module
python3 -m pytest test_loops.py -k multichrm # Runs tests that contains substring 'multichrm' 
```

Following Test Driven Development(TDD) principle, tests should be written whenever new capabilities are added. 

## Additional Information 

If you use `FancBoundary` class in `tad.py`, you'll need to install `hdf5` library and `fanc` package. Check [here](https://vaquerizaslab.github.io/fanc/getting_started.html)