# Hi-C Tools

## HiC Explorer

### Format Conversion
**Convert Format**

First, `hic` to `cool`

```sh
hicConvertFormat --matrices data/GSE151553_A364_merged.hic --outFileName data/GSE151553_A364_merged.cool --inputFormat hic --outputFormat cool 
```

Then, `cool` to `h5`

```sh
hicConvertFormat --matrices data/GSE151553_A364_merged.mcool --outFileName data/GSE151553_A364_merged.h5 --inputFormat mcool --outputFormat h5 
```


**Find TADs**

```sh
hicFindTADs -m data/GSE151553_A364_merged.mcool::/resolutions/500 --outPrefix data/generated_data/chrIX_min1000_max25000_step1500_thres0.05_delta0.01_fdr --chromosomes IX --minDepth 2000 --maxDepth 5000 --step 1000 --thresholdComparisons 0.05  --delta 0.01 --correctForMultipleTesting fdr -p 64
```

## FAN-C

### TAD
**Plot domain**

```sh
fancplot -o figures/chrix_180kb_280kb.png IX:180kb-280kb -p triangular data/GSE151553_A364_merged.juicer.hic@800 -m 50000 -vmin 0 -vmax 50
```

#### Insulation Score

**Find insulation Score**

```sh
fanc insulation data/GSE151553_A364_merged.juicer.hic@500 data/generated_data/chrix_100kb_400kb.insulation -r IX:100kb-400kb -o bed -w 1000 2000 5000 10000 25000 
```

**Plot Insulation Score**

```sh
fancplot --width 6 -o figures/chrix_500b_tads_insulation_1k_2k_5k.png IX:100kb-400kb -p triangular data/GSE151553_A364_merged.juicer.hic@500 -m 50000 -vmin 0 -vmax 50 -p line data/generated_data/chrix_100kb_400kb.insulation_1kb.bed data/generated_data/chrix_100kb_400kb.insulation_2kb.bed data/generated_data/chrix_100kb_400kb.insulation_5kb.bed -l "1kb" "2kb" "5kb"
```

**Find Boundaries**

```sh
fanc boundaries data/generated_data/chrix_100kb_400kb.insulation_1kb.bed data/generated_data/chrix_100kb_400kb.insulation_1kb_boundaries 
```

**Plot Boundaries**

```sh
fancplot --width 6 -o figures/chrix_500b_tads_insulation_boundaries_1k.png IX:100kb-400kb -p triangular data/GSE151553_A364_merged.juicer.hic@500 -m 50000 -vmin 0 -vmax 50 -p line data/generated_data/chrix_100kb_400kb.insulation_1kb.bed -l "1kb" -p bar data/generated_data/chrix_100kb_400kb.insulation_1kb_boundaries
```

[TAD Analysis - FAN-C](https://vaquerizaslab.github.io/fanc/fanc-executable/fanc-analyse-hic/domains.html)

## Juicer

**Run Juicebox**

```sh
java -Xms512m -Xmx2048m -jar Juicebox.jar
```
