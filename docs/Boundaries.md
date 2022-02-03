# Boundaries

Experiments of C0 in boundaries.

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

### Hi-C Explorer
We identify domains and boundaries with Hi-C explorer. A bp is either in a domain or in a boundary. A boundary = a bin. 

Param BH1: Bndry res = 200bp, lim = 100bp, score_perc = 0.5  (chrV 67)
Param BH2: Bndry res = 500bp, lim = 250bp, score_perc = 1.0 

NDR Length 
Method DL1: linker >= 80bp.
Method DL2: linker >= 40bp.

NDR Common
Method DC1: DL1. NDR completely in Bnd.
Method DC2: DL2. NDR ovlps with Bnd >= 40 bp.

### Fan-C
|Param | res(bp) | window size  | min score | Total    | ChrV | Comments | 
|-----|----------|--------------|-----------|----------|------|----------|
|     | 200      | 1000         | 2.0       | 1235/5676| 77   |  |
|     | 200      | 2000         | 2.0       | 650/4830 | 37   |  |
|     | 200      | 2000         | 1.75      | 884/4830 | 60   |  |
|     | 200      | 2000         | 1.5       | 1180/4830| 75   |  |
|     | 200      | 5000         | 1.0       | 710/3815 | 52   | best |
| BF1 | 200      | 5000         | 25%       |    /3815 | 45   |  |
|     | 500      | 1000         | 1.5       | 584/2677 | 36   |  |


## C1. Boundaries and Promoters

### Exp 1.1. Number and Prob distrib of C0 in various comb. of promoters and boundaries

#### Research Questions
1. Any diff in C0 distrib between prmtr bndry and non-prmtr bndry? 
2. Any diff in C0 distrib between prmtr w/ bndry and prmtr w/o bndry?
3. Number 

#### Procedure
Prob. dist / density of C0 in PM, BN, PM/B, BN/P, CH [C3] 

- Box plot: param 1 mean c0[F12]
- Density: param 1 mean c0[F11]
- Density: param 2 mean c0[F4]

#### Observations

##### Param 1 (More imp)
1. Bndrs and prmtr bndrs both show more distrib towards higher C0, stretched bell curve. Non-prm bndrs do not show such skewness, rather even distributed, similar to genes, maybe not bndrs at all. Bndrs are short len (100 bp). 
2. PMoB and PM similar distrib. PMwB are much more bendable in comparison and than genes. Probably because high width (500 bp).   
3. Number: In chr V, Prm bndrs - 42, Nonprm bndrs - 25, Prm w/ bndrs - 54, Prm w/o bndrs - 196

##### Param 2
1. Boundaries, both in and not in promoters, show similar bell-curve distribution. In-promoter ones are more concentrated near avg.
2. PMs in general, have a preference for higher bendability. This preference is more visible in PMoB but not in PMwBs. In majority chromosomes (about 3 out of 5) including actual chr V, PMwBs showed a preference for rigidity.



#### Findings
1. Maybe boundaries have a rigid region for 50-100bp flanked by very bendable region for >200bp. 




### Exp 2.2. Closest TSS Distance from Boundary 

#### Procedure
[C8][F9]

#### Observations
- About 60% strong boundaries are within +- 250 bp of a TSS. 
- Only 20% random locs are within +- 250bp of a TSS.



### Exp 2.3 Can simple dinuc occur and distance explain diff in C0 in PMwB and PMoB?

#### Research Question 
1. Any correlation between C0, TpA content and CpG content?

#### Procedure
Using param 1, mean C0, TpA and CpG content of PMwB and PMoB were checked. [C11, C12][F13, F14] 

#### Observation 
1. TpA or CpG does not show any correlation with Mean C0. For both, pearson's r with mean C0 = 0.09

## C2. Boundaries and Linkers

### Exp 2.1. Number and Comparison of C0 in boundaries w/ or w/o NDRs 

#### Procedure
Boundaries with and without any linkers >= x bp were considered. Number [C4] [F5]
Method A: BH2. DC1.
Method B: BH1. DC2. 
Method C: BF1. DC2. 

#### Observations
*Number*
Method A: With NDRs 22 (40%), without NDRs 37 (60%)
Method B: Bnd With NDRs 45, Bnd w/o NDRs 15. NDR in Bnd 45, NDR out Bnd 785. 
Method C: Bnd w NDRs 32, Bnd w/o NDRs 13. NDR in Bnd 31, NDR out Bnd 807. 

*C0*
?




### Exp 2.2. Closest NDR distance from boundary

#### Procedure
Closest distance from boundary middle to a long linker (>=40 bp) middle was recorded. [C5, C6][F6, F7]
Param - ?

#### Observations
- Significantly higher numbers of linkers >=30 or 40 bp are close to boundaries than linkers >= 60 or 80 bp. Suggests that 40bp linker is a good threshold to find NDRs near boundaries. 
- When only strong boundaries are considered (stronger 2 quartiles), about 90% boundaries have a linker >=40 bp within +- 250 bp. (That's a lot of distance!)

- Only 60% random locs are within +- 250bp of a long linker (>=40bp).

#### Future Directions
- See nearest long linker distance from random positions or whole chromosome.


## C3. Mean C0 in Boundaries

### Exp 3.1. Check mean C0 around boundary region 

**Procedure** 
Take mean of +- 250 bp around boundary mid. [C1] [F1]

**Observations**
- Promoter boundaries: 
  - In most chrms, periodic rigidity and bendability is seen at about one nucleosomal distance. Rigid region is not highly rigid, not exactly at boundary mid. 
  - In big chromosomes (e.g. IV), the dip is not much observable. Because of misalignment of boundaries.

- Non-promoter boundaries: 
  - Usual fluctuation for nucleosomes.
  - Surprisingly, in all chrms, one/two clearly visible rigid region exists.
  

### Exp 3.2. Comparison of mean C0 of TADs and boundaries 

**Procedure** 
Compare mean C0 of all, promoter, non-promoter boundaries and domains in all chromosomes. [C2] [F2]

**Observations**
- Out of 16 chromosomes, In 15, domains are less bendable than boundaries. 
- Out of 16 chromosomes, In all 16, promoter boundaries > domains. In 13, non-promoter boundaries > domains. 

Taekjip Ha group showed that promoter regions are more bendable. It is a known fact that most boundaries reside in promoter regions (~70% in our analysis). So, boundaries are ought to be more bendable. 

### Exp 3.3. Plot mean C0 of a few kbp regions around boundary

**Motivation**
Samin saw a peak in boundary compared to regions in domain. 

**Procedure**
Take mean c0 at each bp for a long segment (6000 bp) around boundary. [C1] [F3] 

**Observations**
- Too many fluctuations to see any pattern.
- C0 at BNnP varies more than BNiP.  
- Sinusoidal pattern shown by Samin, high on left of mid and low on right of mid, is seen at more boundaries (5 / 7 I saw). But not in some, e.g. VI
- Nice peak of C0 when smoothed shown by Samin was observed in raw eyes only at a few BN. ( 2 / 7 I saw)


## C4. Boundary Plots

### Exp 4.1. Inspect long linker positions, length, seq, C0; rigid/flexible region around each boundary by plotting.

#### Research Question
Any pattern in 

*Long Linker*
1. Occur. Prob. of long linkers 
2. length of long linkers. 
3. seq pattern of long linkers. 
4. C0 of long linkers. 

*Sharp Dip / Bendable*
1. Occur prob. of sharp dip / bendable.
2. Width of these regions
3. within nuc / linker?

*Strong vs. weak*
1. Position
2. C0

*Distance from TSS*
1. Position

#### Procedure
Bndry indiv plotted with nuc, linker pos. [C9][F10]
Method A: BH2. DL2.
Method B: BF1. DL2. 

#### Observations
Method A
*Long Linker*
1. In prmtr bndry, >80% times close. In non-prmtr bndry, >60% times close. 
2. \>60% long linkers were very long (>100bp)
3. 30% times runs of A, runs of T. Other times no pattern, specially very long linkers. (prob missing nuc)
4. In promoters, about 50% times downward hill. In non-promoters, about 50% times downward. 

*Sharp Dip/Bendable*
1. 
- In prmtr bndrs, About 60% times sharp dip in or close to bndry mid. 
- In non prmtr bndrs, in those with long linkers, about 70% times sharp dip. In those with nuc, about 30% times sharp dip. 
2. 50-100 bp
3. 80% times sharp dip are within linker. 

*Strong vs. weak*
1. Strong boundaries are close to TSS. Often at promoters. For example, at 322.2k and 342.6k. 
2. About 25% strong do have long linkers + sharp dip. Same for weaks (~-0.4). So, no correlation between sharp dip + strongness. But, >50% strong have long linkers. About 40% weak have long linkers. 

Method B
*Long Linker*
1. About 90% bndry has a long linker in +-300 bp. About 60% in +-200 bp. 
2. About 50% long linker are <80bp. Some very long. 
3. About 70% long linker contain a dip in C0. This kind of dip is observed in 40% neuclosomal region. 

*Sharp dip in C0*
1. About 60% bndry has sharp dip (< -0.5) within +-75bp. Other 40% has high / peak C0 in this area, some unusually high C0 ( > 1.0)
2. Within +-75bp, Sharp dip around 60% times are without linkers.  

*Strong vs. weak*
1. Not much data. Since only top 25%. No correlation seen between C0 and bndry score. 

### Observations
C0: Some 

## Code References
[C1] `conformation.domains.PlotBoundariesHE.line_c0_around()`
[C2] `conformation.domains.MCBoundariesHECollector.plot_scatter_mean_c0()`
[C3] `conformation.domains.PlotBoundariesHE.prob_distrib()`
[C4] `chromosome.crossregions.DistribPlot.num_prmtrs_bndrs_ndrs`
[C5] `chromosome.crossregions.DistribPlot.prob_distrib_bndrs_nearest_ndr_distnc` 
[C6] `chromosome.crossregions.DistribPlot.distrib_cuml_bndrs_nearest_ndr_distnc`
[C8] `chromosome.crossregions.DistribPlot.distrib_cuml_bndrs_nearest_tss_distnc`
[C9] `chromosome.crossregions.LineC0Plot.line_c0_bndrs_indiv_toppings`
[C10] `chromosome.crossregions.LineC0Plot.line_c0_bndrs_indiv_toppings`
[C11] `chromosome.crossregions.PlotPrmtrsBndrs.dinc_explain_box`
[C12] `chromosome.crossregions.PlotPrmtrsBndrs.dinc_explain_scatter`


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
[F10]
![A bndry indiv](../figures/boundaries/VL/bndry_prmtr_82750_83250_score_-0.26_res_200.png)
[F11]
![Bndry prmtr prob distrib c0 res200 lim100](../figures/crossregions/bndrs_prmtrs_prob_distrib_c0_res_200_lim_100_perc_0.5_ustr_500_dstr_-1_VL.png)
[F12]
![Box prmtrs bndrs mean c0](../figures/crossregions/box_bndrs_prmtrs_res_200_lim_100_perc_0.5_ustr_500_dstr_-1.png)
[F13]
![Prmtrs Mean C0, TpA, CpG box](../figures/crossregions/dinc_explain_VL.png)
[F14]
![Prmtrs mean C0, TpA, CpG scatter](../figures/crossregions/prmtrs_ta_cg_scatter.png)
