# Domains

Experiments of C0 in domains.

## Acronyms

BN = Boundaries
PM = Promoters
PMwB = Promoters with boundaries
PMoB = Promoters without boundaries
BN = Boundaries
BNiP = Boundaries in promoters
BNnP = Boundaries not in promoters
CH = Chromosome

## Method

We identify domains and boundaries from 500bp resolution Hi-C data. A bp is either in a domain or in a boundary. A boundary = single pixel = 500 bp. 

We predict bendability of short sequences of all chromosomes in yeast. We then measure bendability of long sequences. 

## Check mean C0 around boundary region 

**Procedure** 
Take mean of +- 250 bp around boundary mid. [C1] [F1]

**Observations**
- Promoter boundaries: 
  - In most chrms, a rigid region than surrounding region. Probably Nucleosome Free Region (NFR). 
  - In big chromosomes (e.g. IV), the dip is not much observable. Because of misalignment of boundaries.

- Non-promoter boundaries: 
  - Usual fluctuation for nucleosomes.
  - Surprisingly, in all chrms, one/two clearly visible rigid region exists.
  

## Comparison of mean C0 of TADs and boundaries 

**Procedure** 
Compare mean C0 of all, promoter, non-promoter boundaries and domains in all chromosomes. [C2] [F2]

**Observations**
- Out of 16 chromosomes, In 15, domains are less bendable than boundaries. 
- Out of 16 chromosomes, In all 16, promoter boundaries > domains. In 13, non-promoter boundaries > domains. 

Taekjip Ha group showed that promoter regions are more bendable. It is a known fact that most boundaries reside in promoter regions (~70% in our analysis). So, boundaries are ought to be more bendable. 

## Plot mean C0 of a few kbp regions around boundary

**Motivation**
Samin saw a peak in boundary compared to regions in domain. 

## Try 1
**Procedure**
Take mean c0 at each bp for a long segment (6000 bp) around boundary. [C1] [F3] 

**Observations**
- Too many fluctuations to see any pattern.
- C0 at BNnP varies more than BNiP.  
- Sinusoidal pattern shown by Samin, high on left of mid and low on right of mid, is seen at more boundaries (5 / 7 I saw). But not in some, e.g. VI
- Nice peak of C0 when smoothed shown by Samin was observed in raw eyes only at a few BN. ( 2 / 7 I saw)

## Probability distribution of C0 in various comb. of promoters and boundaries

**Procedure**
Prob. dist / density of C0 in PM, BN, PM/B, BN/P, CH [C3] [F4]

**Observations**
- Promoters in general, have a preference for higher bendability. This preference is more visible in promoters without boundaries but not in those with boundaries. In majority chromosomes (about 3 out of 5), promoters that form boundaries showed a preference for rigidity. 
- Boundaries, both in and not in promoters, show similar bell-curve distribution. In-promoter ones are more concentrated near avg. 

Observed in chrm VL 


## Code References
[C1] `conformation.domains.PlotBoundariesHE.line_c0_around()`
[C2] `conformation.domains.MCBoundariesHECollector.plot_scatter_mean_c0()`
[C3] `conformation.domains.PlotBoundariesHE.density()`

## Figure References
[F1] 
![Mean c0 around boundaries in chrm III](../figures/domains/mean_c0_bndrs_III.png)
[F2]
![Mean c0 of domains and boundaries in all chrm](../figures/mcdomains/bndrs_dmns_c0_res_200_lim_500_md_30_without_vl.png)
[F3]
![+-3000 bp](../figures/domains/mean_c0_bndrs_IX_plt_3000.png)
[F4]
![Prob dist PM, BN](../figures/domains/boundaries_density_c0_res_500_lim_250_ustr_500_dstr_0_s_mean7_m_None_VL.png)