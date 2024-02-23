
cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo
sleep 5
/app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
   --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240205_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426 \
   --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/random_sets/random_sets_20240202_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426.pkl \
   -hf /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/bands/MLBS_2018_Reflectance_reflectance_warp_scaled.tif \
   -t 20
