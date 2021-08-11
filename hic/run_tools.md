# Hi-C Tools

## HiC Explorer

### Format Conversion

**Convert Format**

First, `hic` to `cool`

Not setting resolution will convert for every resolution. This will create a multiple cool(mcool) file.

```sh
hicConvertFormat --matrices data/GSE151553_A364_merged.hic --outFileName data/GSE151553_A364_merged.mcool --inputFormat hic --outputFormat cool
```

Or, create a cool file with single resolution. 

```sh
hicConvertFormat --matrices data/GSE151553_A364_merged.hic --outFileName data/GSE151553_A364_merged.cool --inputFormat hic --outputFormat cool --resolutions 500
```

**Find TADs**

```sh
hicFindTADs -m data/GSE151553_A364_merged.mcool::/resolutions/500 --outPrefix data/generated_data/IX_res_500_min2000_max5000_step1000_thres0.05_delta0.01_fdr --chromosomes IX --minDepth 2000 --maxDepth 5000 --step 1000 --thresholdComparisons 0.05  --delta 0.01 --correctForMultipleTesting fdr -p 64
```

This will divide whole chromosome into regions(domains). So, there are no non-domain region.

**Plot TADs**

```sh
hicPlotTADs --tracks track.ini -o figures/ix_domains_hicexp.png --region chrIX:1-434000
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
