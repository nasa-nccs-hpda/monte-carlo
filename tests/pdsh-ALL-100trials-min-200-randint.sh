
cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo
sleep 5
/app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
   --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240217_collection-ALL_min_occurrences-200_batch_size-10_num_samples-10897_bandLen-447 \
   --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/random_sets/randint_sets_20240217_collection-ALL_max_occurrences-200_batch_size-10_num_samples-10897_bandLen-447.pkl \
   -hf /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/bands/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif \
   -t 100
