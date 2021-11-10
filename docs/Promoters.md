# Promoters

## Acronyms
PMwD = Promoters with NDR  
PMoD = Promoters without NDR

## Number and C0 of promoters with and without NDR 

**Procedure**
Promoters with and without any linkers >= 80 bp were considered. Number [C2] [F2] and C0 [C1] [F1]

**Observations**  
*Number*  
- With NDRs 100 (40%), without NDRs 150 (60%)

*C0*  
In chr V actual, 
- PMoD are more likely to havehigh bendability - a local hill in higher C0 (-0.05).
- PMwD show typical bell curve. Probably becz low C0 at NDRs and high C0s cancel out. A slight tendency for rigidity - the curve is steeper on high C0 side. 

## Prob distrib of linkers length in promoters 

**Procedure**
Length of linkers completely within promoters. [C3][F3]

**Observations**
- A good amount of prmoters linkers are large. (>= 40bp) 

**Future Directions**
Find percentile of distrib.

## Position of sharp dip of C0 regions in each promoter

**Procedure**
line plot of C0 of each promoter plotted. [C5][F4]

**Observations**
- >90% contains a rigid area (C0 around (-0.6 - -0.8))
- Rigid areas 
  - vary in size - 50bp to 400bp, mostly 200bp.  
  - vary in position. Mostly middle, but sometimes near TSS, sometimes far.  

**Future directions**
Need to know linker positions. 

## Mean and Quartiles of C0 in each promoter

**Procedure**
Promoters data object saved. [C4]

**Observations**
- Out of 250 chromosomes, about half (131 or 52%) has higher C0.
- Are all very rigid promoters near strong boundaries?
  - No. Among 12 promoters with lowest 5% mean C0, 
    - When compared to hi-c visual,
      - 2 or 3 were seen in strong boundaries.
      - most in boundaries, but not strong.
- Are min in quartiles very low?

## Inspect long linker positions, length, seq and C0 in promoters

**Research Question**
- Do long linkers show rigidity?
- Do long linkers have any pattern in seq?
- How long are long linkers?
- Where are long linkers positioned?

**Procedure**

**Observations**


## Code References
[C1] `chromosome.crossregions.CrossRegionsPlot.prob_distrib_prmtr_ndrs`  
[C2] `chromosome.crossregions.CrossRegionsPlot.num_prmtrs_bndrs_ndrs`  
[C3] `chromosome.crossregions.CrossRegionsPlot.prob_distrib_linkers_len_prmtrs`
[C4] `chromosome.regions.Regions.save_regions`
[C5] `chromosome.genes.PlotPromoters.line_c0_indiv`

## Figure References 
[F1] 
![Prmtr NDR C0 prob distrib](../figures/genes/prob_distrib_prmtr_ndrs.png)
[F2]
![Num Prmtrs Bndrs NDRs](../figures/genes/num_prmtrs_bndrs_ndr_V.png)
[F3]
![Linkr len in prmtrs](../figures/linkers/prob_distr_len_prmtrs_V.png)
[F4]
![A promoter C0](../figures/promoters/VL/frw_57812_58311.png)