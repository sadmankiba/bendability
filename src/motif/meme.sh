bnds=1

MINW=8
MAXW=8

PATIENCE=8
THRESH=0.2

GEN_DATA_DIR=/media/sakib/Windows/sakib/programming/playground/machine_learning/bendability/data/generated_data
BND_DIR=boundaries/chrm_s_mcvr_m_None_VL_bf_res_200_lim_100_perc_0.5_fanc
DMN_DIR=domains/chrm_s_mcvr_m_None_VL_dmnsf_bf_res_200_lim_100_perc_0.5_fanc
NUC_BND_DIR=nucleosomes/chrm_s_mcvr_m_None_VL_nucs_w147_bf_res_200_lim_100_perc_0.5_fanc
NUC_DMN_DIR=nucleosomes/chrm_s_mcvr_m_None_VL_nucs_w147_dmnsf_bf_res_200_lim_100_perc_0.5_fanc
BND_SEQ_DIR=${GEN_DATA_DIR}/${BND_DIR}
DMN_SEQ_DIR=${GEN_DATA_DIR}/${DMN_DIR}
NUC_BND_SEQ_DIR=${GEN_DATA_DIR}/${NUC_BND_DIR}
NUC_DMN_SEQ_DIR=${GEN_DATA_DIR}/${NUC_DMN_DIR}

PRIM=${NUC_BND_SEQ_DIR}
CNTR=${NUC_DMN_SEQ_DIR}
streme_cntr() {
    echo "streme cntr"
    streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
        -oc ${PRIM}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}_cnt/ \
        --p ${PRIM}/seq.fasta --n ${CNTR}/seq.fasta
}

if [ "$bnds" = 1 ] ; then
    SEQ_DIR=${BND_SEQ_DIR}
else
    SEQ_DIR=${DMN_SEQ_DIR}
fi

streme_enr() {
    echo "streme enr"
    streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
        -oc ${SEQ_DIR}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}/ \
        --p ${SEQ_DIR}/seq.fasta
}

NMOTIFS=10
enr_meme () {
    meme -dna -revcomp -mod anr -minw ${MINW} -maxw ${MAXW} -nmotifs ${NMOTIFS} -p 4\
        -oc ${SEQ_DIR}/meme_out_n_${NMOTIFS}_mnw_${MINW}_mxw_${MAXW}/ \
        ${SEQ_DIR}/seq.fasta
}


streme_cntr