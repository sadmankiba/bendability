# Loops

## Method

- Predict bendability of sequences of all chromosomes in Yeast using our CNN model. 
- From these short sequences, we measure average bendability of long regions such as within loops or not within loops. 
- Considering all chromosomes, sequences in loop = 80%, in non-loop = 20%. [?]

We list the experiments below:

## Comparison of C0 of sequences within loops and non-loops 

### Exp 1. Mean C0 comparison in each chromosome
#### Research Question
- Are loops more rigid or more bendable? 

#### Procedure 
[C1] [F1]

#### Observation
- Out of 16 chromosomes, In 14 (87.5 %), loop regions are less bendable than non-loop regions.

### Exp 2. Histogram of mean C0 of all loops and non-loops in all chromosomes
#### Research Question 
- Does non-loop show more bendability? 

#### Procedure 
[C2] [F2]

#### Observation
- Non-loops are slightly right (more bendable) of loops.

## FAQ 
**Q** How reliable is the 500 res loops? 
- looks very good. The few anchors I looked at were all anchors in the hi-c map. So, I guess it's about 80% correct. 


## Code References 
[C1] `conformation.meanloops.MultiChrmMeanLoopsCollector.plot_scatter_loop_nuc_linker_mean()`
[C2] `conformation.coverloops.PlotMCCoverLoops.plot_histogram_c0()`

## Figure References 
[F1] ![Mean c0 all chrm](../figures/mcloops/nuc_linker_mean_md_30_mx_None_without_vl.png)
[F2] ![Double hist](../figures/mcloops/hist_c0_bins_40_all_pred.png)
