# Motifs 

## Base work
Motif M30
**Contribution score**: We determined motif logo and contribution score for each motif. Contrib score ranged from -0.017 to 0.021. The higher the contrib score, the greater that motif contribute to positive bendability. [F1]

5 +ve contrib motifs: TATA, GAAGA, CCCT, TGCA, CCTT
5 -ve contrib motifs: TTTTT, AAAA, TTGG, GAAGA, CGCG

Motif M35
## Experiments 

### Exp 1.3 Motif enrichment in boundaries
#### Research question
- Are any motifs, provided by model, enriched in boundaries compared to domain or whole chromosome?

#### Procedure 
Motif M35.
- Box plot: Distrib of scores of all motifs in regions. [C3][F3, F4]
- Z-test: Z-test of distribution of scores in boundaries and domains. [C4] [D1]
#### Observation
BoundariesF
- Box plot: no observable difference in score distribution between boundaries and whole chromosome. 
- We find that boundaries are mostly characterized by enrichment of certain motifs (mostly long run of A or T). On the other hand, a few motifs are de-enriched in boundaries. They typically contain C/G dinucleotide or trinucleotides.

#### Future Direction 
- Find mathematically which motifs have highest difference in score distributions between boundary and whole chromosome. 
- Compare between boundaries and domains.

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
[C3] `motif.motifs.MotifsM35.enrichment`
[C4] `motif.motifs.MotifsM35.enrichment_compare`

## Data References
[D1] [motif enrichment](../data/generated_data/boundaries/motif_m35/enrichment_comp_res_200_lim_100_perc_0.5_fanc_domains_res_200_lim_100_perc_0.5_fanc.tsv)

## Figure References 
[F1] [Motif logos](../figures/motifs/motif_logos_sorted_contrib_with_motif_no.png)
[F2] [Motif contrib to PMwB and PMoB](../figures/promoters/distribution_around_promoters/both_sorted_motif/motif_179_188.png)
[F3] [Boundary motif enrichment](../figures/boundaries/motif_m35/enrichment_res_200_lim_100_perc_0.5_fanc.png)
[F4] [Chrm motif enrichment](../figures/chromosome/motif_m35/enrichment_regions.png)