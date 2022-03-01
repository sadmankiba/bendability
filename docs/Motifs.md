# Motifs 

## Base work
**Contribution score**: We determined motif logo and contribution score for each motif. Contrib score ranged from -0.017 to 0.021. The higher the contrib score, the greater that motif contribute to positive bendability. [F1]

5 +ve contrib motifs: TATA, GAAGA, CCCT, TGCA, CCTT
5 -ve contrib motifs: TTTTT, AAAA, TTGG, GAAGA, CGCG

## Experiments 

### Exp 1.3 Motif enrichment in boundaries
#### Research question
- Are any motifs that is provided by model enriched in boundaries compared to whole chromosome?

#### Procedure 

#### Observation

### Exp 1.2 Motif contribution comparison of non-bndry and bndry NDRs

#### Research Question 
- Does any motif show opposite pattern? 

#### Procedure 
BF1. DL2. Kmer count. [C2]

#### Observations 

### Exp 1.1 Motif contribution comparison in PMwB and PMoB

#### Research Question
1. Does any motif show completely opposite contrib?
2. Any pattern among motifs with highest -ve or +ve contrib to c0?

#### Procedure 
Merged in subplot motif contribs with these sorted by contrib score ascendingly. [C1] [F1] [F2]

#### Observations 
1. No. 
2. 
  a. Motifs with -ve contrib to C0
    i. Most (70%) motifs show higher contrib to without boundaries. Others show similar pattern in both. (expected)
  
  b. Motifs with +ve contrib to c0 
    i. No observable preference found for PMwB. (unexpected)
    ii. Some (30%) showed slight preference for PMwB. Most on left of promoter mid. 
    iii. Surprisingly, some (30%) showed preference for PMoB. Most on right of promoter mid. 
  
  c. Midrange
    i. Similar in both. (expc.)

##### Comments
Expected that positive motif will be enriched at PMwB since they were found to be more bendable. 

#### Future Directions 
1. Align promoters according to gene direction.
2. Cluster motifs by contribution patterns in PMwB and PMoB. Simplest, pearson.


## Code References
[C1] `chromosome.crossregions.PlotPrmtrsBndrs.both_sorted_motif_contrib`
[C2] `chromosome.crossregions.ScatterPlot.scatter_kmer`

## Figure References 
[F1] [Motif logos](../figures/motifs/motif_logos_sorted_contrib_with_motif_no.png)
[F2] [Motif contrib to PMwB and PMoB](../figures/promoters/distribution_around_promoters/both_sorted_motif/motif_179_188.png)