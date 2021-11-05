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
- PMs in general, have a preference for higher bendability. This preference is more visible in PMoB but not in PMwBs. In majority chromosomes (about 3 out of 5) including actual chr V, PMwBs showed a preference for rigidity. 
- Boundaries, both in and not in promoters, show similar bell-curve distribution. In-promoter ones are more concentrated near avg. 

 

## Number and Comparison of C0 in boundaries w/ or w/o NDRs 

**Procedure**
Promoters with and without any linkers >= x bp were considered. Number [C4] [F5]

**Observations**
*Number*
- Bndry: res 500bp, width 500bp, score perc: 1 linker: >= 80bp: 
  - With NDRs 22 (40%), without NDRs 37 (60%)
- Bndry: res 200bp, width 500bp, score perc: 0.5 linker: >= 40bp:
  - 
  
*C0*
?

## Closest NDR distance from boundary

**Procedure**
Closest distance from boundary middle to a long linker (>=40 bp) middle was recorded. [C5, C6][F6, F7]

**Observations**
- Significantly higher numbers of linkers >=30 or 40 bp are close to boundaries than linkers >= 60 or 80 bp. Suggests that 40bp linker is a good threshold to find NDRs near boundaries. 
- When only strong boundaries are considered (stronger 2 quartiles), about 90% boundaries have a linker >=40 bp within +- 250 bp. 
- Only 60% random locs are within +- 250bp of a long linker (>=40bp).

**Future Directions**
- See nearest long linker distance from random positions or whole chromosome.

## Plot c0, boundary, nuc and tss position on same plot

**Procedure**
Plot c0 with imp pos. [C7][]

**Observations**
- Are boundaries near rigid linkers?
  - From quick glance, no such relation was found.

- How are boundaries distanced from TSS?
  - Strong boundaries are close to TSS. Often at promoters. For example, at 322.2k and 342.6k. 

- Any pattern in C0 that differentiates strong boundaries to weak boundaries? 

**Future directions**
- TSS distnc from bndry prob distrib
- C0 region around bndry 

## Closest TSS Distance from Boundary 

**Procedure**
[C8][F9]

**Observations**
- About 60% strong boundaries are within +- 250 bp of a TSS. 
- Only 20% random locs are within +- 250bp of a TSS.

## Code References
[C1] `conformation.domains.PlotBoundariesHE.line_c0_around()`
[C2] `conformation.domains.MCBoundariesHECollector.plot_scatter_mean_c0()`
[C3] `conformation.domains.PlotBoundariesHE.prob_distrib()`
[C4] `chromosome.crossregions.CrossRegionsPlot.prob_distrib_linkers_len_prmtrs`
[C5] `chromosome.crossregions.CrossRegionsPlot.prob_distrib_bndrs_nearest_ndr_distnc` 
[C6] `chromosome.crossregions.CrossRegionsPlot.distrib_cuml_bndrs_nearest_ndr_distnc`
[C7] `chromosome.crossregions.CrossRegionsPlot.line_c0_toppings`
[C8] `chromosome.crossregions.CrossRegionsPlot.distrib_cuml_bndrs_nearest_tss_distnc`

## Figure References
[F1] 
![Mean c0 around boundaries in chrm III](../figures/domains/mean_c0_bndrs_III.png)
[F2]
![Mean c0 of domains and boundaries in all chrm](../figures/mcdomains/bndrs_dmns_c0_res_200_lim_500_md_30_without_vl.png)
[F3]
![+-3000 bp](../figures/domains/mean_c0_bndrs_IX_plt_3000.png)
[F4]
![Prob dist PM, BN](../figures/domains/boundaries_prob_distrib_c0_res_500_lim_250_ustr_500_dstr_0_s_mean7_m_None_VL.png)
[F5]
![Num Prmtrs Bndrs NDRs](../figures/genes/num_prmtrs_bndrs_ndr_V.png)
[F6]
![Prob distrib bndry nearest NDR](../figures/boundaries/distnc_ndr_prob_distrib_res_500_V.png)
[F7]
![Cumulative percentage bndry nearest NDR](../figures/boundaries/distnc_ndr_distrib_cuml_res_200_perc_0.5_40_V.png)
[F8] 
![C0 with imp pos](../figures/crossregions/line_c0_toppings_339k_345k.png)
[F9]
![Closest TSS from bndry](../figures/boundaries/distnc_tss_distrib_cuml_res_200_perc_0.5_V.png)