if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("DNAshapeR")

library(DNAshapeR)

fn <- "/home/sakib/playground/machine_learning/bendability/data/rl.fasta"
pred <- getShape(fn)

pred["Roll"]


