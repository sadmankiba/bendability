# Loops

## Experiments

**Methods**

- Predict bendability of sequences of all chromosomes in Yeast using our CNN model. 
- From these short sequences, we measure average bendability of long regions such as within loops or not within loops. 
- Considering all chromosomes, sequences in loop = 80%, in non-loop = 20%. [?]

We list the experiments with research questions as title.

### Compare C0 of sequences within loops and non-loops 

#### Mean C0 comparison in each chromosome

`conformation.meanloops.MultiChrmMeanLoopsCollector.plot_scatter_loop_nuc_linker_mean()`

![mean c0 all chrm](../figures/mcloops/nuc_linker_mean_md_30_mx_None.png)

Out of 16 chromosomes, In 14 (87.5 %), loop regions are less bendable than non-loop regions.

#### Mean C0 comparison of all loops vs. non-loops in all chromosomes


- Among total x loops, m loops (?) are less bendable than seq outside loops. [?]

#### Histogram of mean C0 of all loops and non-loops in all chromosomes

`conformation.coverloops.PlotMCCoverLoops.plot_histogram_c0()`

![Double hist](../figures/mcloops/hist_c0_bins_40_all_pred.png)

- Non-loops are slightly right of loops.

