
cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
#\cp -r /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/random_sets_20240111_shrink_False_num_samples_1088_bandLen_447.bak /explore/nobackup/projects/ilab/data/AGB/test/mlruns/random_sets/random_sets_20240111_shrink_False_num_samples_1088_bandLen_447 \
/panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo

/app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
   --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/202410130-gpu004-40cpus-callbacks-divideby10k_noMinMax \
   --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mlruns/random_sets/random_sets_20240111_shrink_False_num_samples_1088_bandLen_447 \
   -hf /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif \
   -t 300 \
   --shapArchive /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/202410130-gpu004-40cpus-callbacks-divideby10k_noMinMax/225919640119481430/PERMUTATION_IMPORTANCE_VALUES/MLP_SGD_7_layers_1024_units_MAE_50_epochs_callbacks::225919640119481430.keras


