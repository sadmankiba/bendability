RES=800
DATA=data/GSE151553_A364_merged.juicer.hic 
CHROMOSOME=IV
START_KB=180
END_KB=280
WINDOW_KB=1


fancplot -o figures/chr${CHROMOSOME}_${START_KB}kb_${END_KB}kb_fanc.png \
    ${CHROMOSOME}:${START_KB}kb-${END_KB}kb -p triangular ${DATA}@${RES} -m 50000 -vmin 0 -vmax 50

INSULATION_FILE=data/generated_data/chr${CHROMOSOME}_${START_KB}kb_${END_KB}kb.insulation 
fanc insulation ${DATA}@${RES} ${INSULATION_FILE} \
    -r ${CHROMOSOME}:${START_KB}kb-${END_KB}kb -o bed -w 1000 2000 5000 10000 25000
fanc boundaries ${INSULATION_FILE}_${WINDOW_KB}kb.bed ${INSULATION_FILE}_${WINDOW_KB}kb_boundaries
