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

**Future Directions**
Find percentile of distrib.

## Code References
[C1] `chromosome.crossregions.CrossRegionsPlot.prob_distrib_prmtr_ndrs`
[C2] `chromosome.crossregions.CrossRegionsPlot.num_prmtrs_bndrs_ndrs`
[C3] `chromosome.crossregions.CrossRegionsPlot.prob_distrib_linkers_len_prmtrs`

## Figure References 
[F1] 
![Prmtr NDR C0 prob distrib](../figures/genes/prob_distrib_prmtr_ndrs.png)
[F2]
![Num Prmtrs Bndrs NDRs](../figures/genes/num_prmtrs_bndrs_ndr_V.png)
[F3]
![Linkr len in prmtrs](../figures/linkers/prob_distrib_len_prmtrs_V.png)