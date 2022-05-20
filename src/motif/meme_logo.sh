# Generate a motif logo from a meme-suite motif file

GEN_DATA_DIR=/media/sakib/Windows/sakib/programming/playground/machine_learning/bendability/data/generated_data
BND_DIR=chrm_s_mcvr_m_None_VL_bf_res_200_lim_100_perc_0.5_fanc
DMN_DIR=chrm_s_mcvr_m_None_VL_dmnsf_bf_res_200_lim_100_perc_0.5_fanc
BND_SEQ_DIR=${GEN_DATA_DIR}/boundaries/${BND_DIR}
DMN_SEQ_DIR=${GEN_DATA_DIR}/domains/${DMN_DIR}
MEME_DIR=${DMN_SEQ_DIR}/streme_out_mnw_6_mxw_15_pt_8_th_0.2_cnt

MOTIF_NO=6
ceqlogo -i${MOTIF_NO} ${MEME_DIR}/streme.txt -f PNG -d "" -o ${MEME_DIR}/logos/${MOTIF_NO}.png