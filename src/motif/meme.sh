GEN_DATA_DIR=/media/sakib/Windows/sakib/programming/playground/machine_learning/bendability/data/generated_data
BND_DIR=chrm_s_mcvr_m_None_VL_bf_res_200_lim_100_perc_0.5_fanc
NMOTIFS=10
MINW=6
MAXW=20

meme -dna -revcomp -mod anr -minw ${MINW} -maxw ${MAXW} -nmotifs ${NMOTIFS} -p 4\
    -oc ${GEN_DATA_DIR}/boundaries/${BND_DIR}/meme_out_n_${NMOTIFS}_mnw_${MINW}_mxw_${MAXW}/ \
    ${GEN_DATA_DIR}/boundaries/${BND_DIR}/seq.fasta 