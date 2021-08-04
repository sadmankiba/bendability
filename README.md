# DNA Bendability Analysis

This repository contains various utility tools to analyze DNA Bendability. 

## Data Used 

1. Measured Intrinsic Cyclizability from [Measuring DNA mechanics on the genome scale](https://www.nature.com/articles/s41586-020-03052-3) paper. 

2. 3D Genomic Organization of Yeast found with Micro-C in [Cohesin residency determines chromatin loop patterns](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE151553) paper. 

3. Nucleosome center positions from [A map of nucleosome positions in yeast at base-pair resolution](https://www.nature.com/articles/nature11142?page=3)

4. Reference genome sequence of S. Cerevisiae downloaded from [Saccharomyces Genome Database (SGD)](https://www.yeastgenome.org/)


## Using Code 

### Installing Packages

`fanc` package requires `hdf5` library. Check [here](https://vaquerizaslab.github.io/fanc/getting_started.html)

Now, install necessary packages. 

```sh 
pip install -r requirements.txt
```

### Running a function
Invoking functionalities is very simple. Just create approriate object and call it's public functions from `main.py`.

```py
Loops().plot_c0_vs_total_loop(200, 'predicted')
```

Then, run `main.py`. 

```sh
python3 main.py
```

## Testing 

Test if all functionalities are working fine.

```sh
python3 -m pytest
```

Or, run a single test file. 

```sh 
python3 -m pytest test_reader.py
```
