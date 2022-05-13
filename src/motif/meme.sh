RUN_MEME=0
RUN_STREME=1
bnds=1
cnt_dmn=0

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


NMOTIFS=10

if [ "$RUN_MEME" = 1 ] ; then
    meme -dna -revcomp -mod anr -minw ${MINW} -maxw ${MAXW} -nmotifs ${NMOTIFS} -p 4\
        -oc ${SEQ_DIR}/meme_out_n_${NMOTIFS}_mnw_${MINW}_mxw_${MAXW}/ \
        ${SEQ_DIR}/seq.fasta
fi

PATIENCE=8
THRESH=0.2

if [ "$RUN_STREME" = 1 ] ; then
    if [ "$cnt_dmn" = 1 ] ; then
        echo "cnt dmn"
        streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
            -oc ${BND_SEQ_DIR}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}_cnt/ \
            --p ${BND_SEQ_DIR}/seq.fasta --n ${DMN_SEQ_DIR}/seq.fasta
    else
        echo "not cnt dmn"
        streme --dna --minw ${MINW} --maxw ${MAXW} --patience ${PATIENCE} --thresh ${THRESH}  \
            -oc ${SEQ_DIR}/streme_out_mnw_${MINW}_mxw_${MAXW}_pt_${PATIENCE}_th_${THRESH}/ \
            --p ${SEQ_DIR}/seq.fasta
    fi
fi