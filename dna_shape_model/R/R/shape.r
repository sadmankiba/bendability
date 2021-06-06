if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("DNAshapeR")

library(DNAshapeR)

fn <- "/home/samin/workspace/research_implementation_scOmics/regression_model_meuseum_mod/data/sequences_rc_8.fasta"
pred <- getShape(fn)

pred["Roll"]


