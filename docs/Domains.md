# Domains

## Experiments

**Method**

We identify domains and boundaries from 500bp resolution Hi-C data. A bp is either in a domain or in a boundary. A boundary = single pixel = 500 bp. 

We predict bendability of short sequences of all chromosomes in yeast. We then measure bendability of long sequences. 


## Comparison of mean C0 of TADs and boundaries 

`conformation.tad.MCBoundariesHECollector.plot_scatter_mean_c0()`

![Mean c0 of tads and boundaries in all chrm](../figures/mcdomains/bndrs_dmns_c0_res_200_lim_500_md_30_without_vl.png)

- Out of 16 chromosomes, In 15, domains are less bendable than boundaries. 
- Out of 16 chromosomes, In all 16, promoter boundaries > domains. In 13, non-promoter boundaries > domains. 

Explanation: Taekjip Ha group showed that promoter regions are more bendable. It is a known fact that most boundaries reside in promoter regions (~70% in our analysis). So, boundaries are ought to be more bendable. 