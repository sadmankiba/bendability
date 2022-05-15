bnds=1

GEN_DATA_DIR=/media/sakib/Windows/sakib/programming/playground/machine_learning/bendability/data/generated_data
BND_DIR=chrm_s_mcvr_m_None_VL_bf_res_200_lim_100_perc_0.5_fanc
DMN_DIR=chrm_s_mcvr_m_None_VL_dmnsf_bf_res_200_lim_100_perc_0.5_fanc
BND_SEQ_DIR=${GEN_DATA_DIR}/boundaries/${BND_DIR}
DMN_SEQ_DIR=${GEN_DATA_DIR}/domains/${DMN_DIR}

if [ "$bnds" = 1 ] ; then
    SEQ_DIR=${BND_SEQ_DIR}
else
    SEQ_DIR=${DMN_SEQ_DIR}
fi

MINW=6
MAXW=15


PATIENCE=8
THRESH=0.2

enr_streme() {
    echo "enr streme"
    streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
        -oc ${SEQ_DIR}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}/ \
        --p ${SEQ_DIR}/seq.fasta
}

streme_bnd_cntr() {
    echo "streme bnd cntr"
    streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
        -oc ${BND_SEQ_DIR}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}_cnt/ \
        --p ${BND_SEQ_DIR}/seq.fasta --n ${DMN_SEQ_DIR}/seq.fasta
}

streme_dmn_cntr() {
    echo "streme dmn cntr"
    streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
        -oc ${DMN_SEQ_DIR}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}_cnt/ \
        --p ${DMN_SEQ_DIR}/seq.fasta --n ${BND_SEQ_DIR}/seq.fasta
}

NMOTIFS=10

enr_meme () {
    meme -dna -revcomp -mod anr -minw ${MINW} -maxw ${MAXW} -nmotifs ${NMOTIFS} -p 4\
        -oc ${SEQ_DIR}/meme_out_n_${NMOTIFS}_mnw_${MINW}_mxw_${MAXW}/ \
        ${SEQ_DIR}/seq.fasta
}


streme_dmn_cntr