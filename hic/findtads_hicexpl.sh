date_dir="`date +%F`"
time_dir="`date +%H_%M`"

bndry_dir=./data/generated_data/boundaries/$date_dir/$time_dir
mkdir -p $bndry_dir

min_depth=1000
max_depth=10000
step=1000
thres_comparison=0.05
delta=0.01 
correct_for_multiple_testing=fdr

config_file=$bndry_dir/config.txt
echo "min_depth=${min_depth}" >> $config_file 
echo "max_depth=${max_depth}" >> $config_file
echo "step=${step}" >> $config_file 
echo "thres_comparison=${thres_comparison}" >> $config_file
echo "delta=${delta}" >> $config_file 
echo "correct_for_multiple_testing=${correct_for_multiple_testing}" >> $config_file

res=200

for chrm in I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI
do
   hicFindTADs -m data/GSE151553_A364_merged.mcool::/resolutions/${res} \
    --outPrefix ${bndry_dir}/${chrm}_res_${res}_hicexpl \
    --chromosomes ${chrm} --minDepth ${min_depth} --maxDepth ${max_depth} --step ${step} \
    --thresholdComparisons ${thres_comparison}  --delta ${delta} \
    --correctForMultipleTesting ${correct_for_multiple_testing} -p 64
done
