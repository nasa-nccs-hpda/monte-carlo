
cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo
echo /app/jupyter/ilab/tensorflow-kernel-bak/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
    --config ./tests/exp_aggregate_hyperspectral_50epochs_noMinMax.json -o $1 \
   --bandListFile  $2 \
   -hf /explore/nobackup/projects/ilab/data/AGB/test/beta_pmm/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0/MLBS_2018_hyperspectral_indices.tif  \
   -t $3 --prune $4
echo 
/app/jupyter/ilab/tensorflow-kernel-bak/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
    --config ./tests/exp_aggregate_hyperspectral_50epochs_noMinMax.json -o $1 \
   --bandListFile  $2 \
   -hf /explore/nobackup/projects/ilab/data/AGB/test/beta_pmm/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0/MLBS_2018_hyperspectral_indices.tif  \
   -t $3 --prune $4
