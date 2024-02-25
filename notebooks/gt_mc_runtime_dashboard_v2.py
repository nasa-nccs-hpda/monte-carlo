#!/usr/bin/env python
# coding: utf-8

# # Runtime Dashboard

# In[1]:


import shap
import numpy as np
import pickle
import glob
import pandas as pd


# In[2]:


import sys
sys.path.append('/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion')
sys.path.append('/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo')

last0=0
last1=0
last2=0
last3=0
last4=0
last5=0


# In[ ]:


last6=0
last7=0
last8=0
last9=0
last10=0
last11=0


# ## Check status of runs

# ## Bottom line as of 02/03/2024 - Trying runs of 447 bands with minimum of 50 occurrences on new dedicated ilab213 [30 CPUs & 240GB RAM].  Seemingly blew out gpu007 on 2/2/24 too. 

# ## -----------------------------------------------------------------
# ## COMPLETE runs
# ## -----------------------------------------------------------------

# # *FINISHED NEON:* Generate shap files for NEON - 426 - bands @ 20 tasks [10 indices per subset]/1 node [30 CPUs] /60 minutes
# #### Feb  5 14:09 -> Feb  7 18:50 = 52hrs 41m
# #### (base) gtamkin@ilab209: pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-VI-20trials-ilab213.sh &
# #### */10 * * * *  pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-VI-20trials-ilab213.sh 
# #### 10 Min Occurrences - __UNLIMITED__ CRON/10 minutes: pdsh on ilab213

# In[ ]:


# /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240205_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426

#*/10 * * * *  pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-NEON-20trials.sh 


# (base) gtamkin@ilab213:/home/gtamkin$ more /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-NEON-20trials.sh

# cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
# /panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
# export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo
# sleep 5
# /app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
#    --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240205_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426 \
#    --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/random_sets/random_sets_20240202_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426.pkl \
#    -hf /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/bands/MLBS_2018_Reflectance_reflectance_warp_scaled.tif \
#    -t 20

from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240205_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426'
#!ls -alRt $results_dir_cpu/855815048022458276/MODELS/ | grep '\]\.model:' | wc -l
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last5,']')
print('# shap files found so far....[',len(shap_files),']')
last5 = len(shap_files)


# # *FINISHED VIs:* Determine how long it will take @ 20 tasks [10 indices per subset]/1 node [30 CPUs] /60 minutes
# #### (base) gtamkin@ilab209: pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-VI-20trials-ilab213.sh &
# #### */10 * * * *  pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-VI-20trials-ilab213.sh 
# #### 10 Min Occurrences - __UNLIMITED__ CRON/10 minutes: pdsh on ilab213

# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_ilab213_20240205_20trials_collection-VI_max_occurrences-10_batch_size-10_num_samples-128_bandLen-447'
get_ipython().system("ls -alRt $results_dir_cpu/855815048022458276/MODELS/ | grep '\\]\\.model:' | wc -l")
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last0,']')
print('# shap files found so far....[',len(shap_files),']')
last0 = len(shap_files)


# # FINISHED: VI on ilabs202-212 - ~3 hours
# #### */60 * * * *  pdsh -w ilab[202,203,204,205,206,207,208,210,211,212] sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-aggregate-20trials.sh
# 

# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240204_20trials_collection-VI_max_occurrences-10_batch_size-10_num_samples-128_bandLen-447'
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last1,']')
print('# shap files found so far....[',len(shap_files),']')
last1 = len(shap_files)


# # PRUNE FINISHED: ALL on ilab201-212 - min 50 occurrences: ~48 hours
# ## goal = 3227[ilab201-212] -> final 3227 - ~48 hours

# In[ ]:


#goal = 3227[ilab201-212] -> final 3163 - ~48 hours

# drwxr-s---.    6 gtamkin ilab    4096 Feb  9 08:20 af013048e991498b820aad1e774ee8ef ->
#-rw-r-----.    1 gtamkin ilab   20296 Feb 11 08:00 MLP_SGD_7_layers_1024_units_MAE_50_epochs_callbacks::961551132591233530.keras[[55, 11, 430, 100, 309, 135, 6, 134, 447, 381]].shap_values0to50

# # of shap files to process in this run: 0
# Processing complete.  See: /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240209_10trials_collection-ALL_min_occurrences-50_batch_size-10_num_samples-3227_bandLen-447

# 10-59/15 * * * * pdsh -w ilab201 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 15-59/15 * * * * pdsh -w ilab202 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 20-59/60 * * * * pdsh -w ilab203 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 25-59/15 * * * * pdsh -w ilab204 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 30-59/15 * * * * pdsh -w ilab205 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 35-59/15 * * * * pdsh -w ilab206 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 40-59/15 * * * * pdsh -w ilab207 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 45-59/15 * * * * pdsh -w ilab208 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 50-59/15 * * * * pdsh -w ilab209 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 55-59/15 * * * * pdsh -w ilab210 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 00-59/60 * * * * pdsh -w ilab211 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh
# 05-59/15 * * * * pdsh -w ilab212 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-10trials-min-50.sh

results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240209_10trials_collection-ALL_min_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'
#results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240208_20trials_collection-ALL_min_occurrences-100_batch_size-10_num_samples-5658_bandLen-447'
from datetime import datetime
start_time = datetime.now()
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
print('# shap files found on last check ....[',last10,']')
print('# shap files found so far....[',len(shap_files),']')
time_elapsed = datetime.now() - start_time

# !ls -alRt $results_dir_cpu/MODELS/ | grep '\]\.model:' | wc -l
# !ls -alRt $results_dir_cpu/MODELS/ | grep '\]\.test' | wc -l
# !ls -alRt $results_dir_cpu/PERMUTATION_IMPORTANCE_VALUES/ | grep shap |wc -l


print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))
last10 = len(shap_files)


# # DONE AS CAN BE: ALL on ilabs213 - min 100 occurrences: ~60 hours
# ## goal = 5658[ilab201-212] -> final 5650 - ~48 hours
# ### #NOTE:  Accidentally removed index file, so can't reproduce.  Arggh!  Re-running with randint()

# In[ ]:


# goal = 5658[ilab201-212] -> final 5650 - ~48 hours

# drwxr-s---.    6 gtamkin ilab    4096 Feb  8 16:11 98681a5edc3a46aa8c151bfc0dd5bc6e ->
#-rw-r-----.    1 gtamkin ilab   20296 Feb 11 01:35 MLP_SGD_7_layers_1024_units_MAE_50_epochs_callbacks::111962329121030626.keras[[117, 252, 48, 125, 353, 347, 95, 37, 119, 104]].shap_values0to50
#NOTE:  Accidentally removed index file, so can't reproduce.  Arggh!  Re-running with randint()

# */10 * * * *  pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-20trials-min-100.sh 
#
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240208_20trials_collection-ALL_min_occurrences-100_batch_size-10_num_samples-5658_bandLen-447'
from datetime import datetime
start_time = datetime.now()
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last9,']')
print('# shap files found so far....[',len(shap_files),']')
last9 = len(shap_files)


# # FINISHED: VI on ilab213 - min 100 occurrences: ~3 hours, 100 trials per run
# ### Goal = 261, Started @ 2/13 ~5 PM
#Time elapsed on node [ilab213] = (hh:mm:ss.ms) 1:04:21.295479  (for 100)
# In[ ]:


get_ipython().system("ls -alRt /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/ilab213-finale-VI/681206886322137043/MODELS/ | grep '\\]\\.model:' | wc -l")
get_ipython().system("ls -alRt /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/ilab213-finale-VI/681206886322137043/MODELS/ | grep '\\]\\.test_results' | wc -l")
get_ipython().system('ls -alt /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/ilab213-finale-VI/681206886322137043/PERMUTATION_IMPORTANCE_VALUES/*shap_* | wc -l')
#!ls -alRt $results_dir_cpu/MODELS/ | grep '\]\.model:' |


# # FINISHED: NEON on ilab202-212 - min 100 occurrences: ? hours 
# ### Goal = 5876, Started @ 2/11 ~11:15AM
### 00-59/15 * * * * pdsh -w ilab202 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-NEON-10trials-min-100-randint.sh
### . . .  
### 50-59/15 * * * * pdsh -w ilab212 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-NEON-10trials-min-100-randint.sh
# In[ ]:


# from datetime import datetime
# start_time = datetime.now()
# results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-NEON_max_occurrences-100_batch_size-10_num_samples-5876_bandLen-426'
# shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
# time_elapsed = datetime.now() - start_time
# print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

# print('# shap files found on last check ....[',last3,']')
# print('# shap files found so far....[',len(shap_files),']')
# last3 = len(shap_files)
get_ipython().system('ls -alt /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-NEON_max_occurrences-100_batch_size-10_num_samples-5876_bandLen-426/286420834988853426/PERMUTATION_IMPORTANCE_VALUES/*shap_* | wc -l')


# # FINISHED: ALL on ilab213 - min 100 occurrences: ? hours 
# ### Goal = 5737, Started @ 2/11 ~10:50AM
# */10 * * * *  pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-ALL-20trials-min-100-randint.sh 
# In[5]:


results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-ALL_max_occurrences-100_batch_size-10_num_samples-5737_bandLen-447/218557492334736085'
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.model:' | wc -l")
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.test' | wc -l")
get_ipython().system('ls -alRt $results_dir_cpu/PERMUTATION_IMPORTANCE_VALUES/ | grep shap |wc -l')


# # FINISHED: NEON on gpu004 - min 100 occurrences: ? hours, min 100 occurences
# ### Goal = 5666, Started @ 2/11 ~10:50AM

# In[9]:


results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240215_collection-NEON_max_occurrences-100_batch_size-10_num_samples-5666_bandLen-426/188577871478003488'
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.model:' | wc -l")
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.test' | wc -l")
get_ipython().system('ls -alRt $results_dir_cpu/PERMUTATION_IMPORTANCE_VALUES/ | grep shap |wc -l')


# ## -----------------------------------------------------------------
# ## IN_PROGRESS runs
# ## -----------------------------------------------------------------

# # GOING: ALL on gpu004, etc. - min 200 occurrences: ? hours
# ### Goal = 10897, Started @ 2/17 ~10:00AM

# In[12]:


results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240217_collection-ALL_min_occurrences-200_batch_size-10_num_samples-10897_bandLen-447/866122034458507573'
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.model:' | wc -l")
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.test' | wc -l")
get_ipython().system('ls -alRt $results_dir_cpu/PERMUTATION_IMPORTANCE_VALUES/ | grep shap |wc -l')


# ## -----------------------------------------------------------------
# ## Valid, but old or partial jobs, before adding filtering per shap file across outputs [Deprecated]
# ## -----------------------------------------------------------------

# # PAUSED: VI on ilab201 - min 100 occurrences: ? hours, restarted 2/12 ~8:20 AM
# ## Goal = 261, Started @ 2/11 ~10:50AM

# In[ ]:


get_ipython().system('ls /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-VI_max_occurrences-100_batch_size-10_num_samples-261_bandLen-21.pk/432327967241936724/PERMUTATION_IMPORTANCE_VALUES/*shap_* | wc -l')

### */10 * * * *  pdsh -w adaptjh2 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-VI-20trials-min-100-randint.sh
# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-VI_max_occurrences-100_batch_size-10_num_samples-261_bandLen-21.pk'
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last1,']')
print('# shap files found so far....[',len(shap_files),']')
last1 = len(shap_files)


# # PAUSED: VI on gpu002 CLI - min 100 occurrences: 1/2 hours 
# ## Goal = 261, Started @ 2/11 ~12:33PM
# ## Time elapsed on node [gpu002] = (hh:mm:ss.ms) 0:32:02.125094 per CLI run
# /app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
#    --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-VI_max_occurrences-100_batch_size-10_num_samples-261_bandLen-21 \
#    --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/random_sets/randint_sets_20240211_collection-VI_max_occurrences-100_batch_size-10_num_samples-261_bandLen-21.pkl \
#    -hf /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/bands/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif \
#    -t 40
# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/randint_sets_20240211_collection-VI_max_occurrences-100_batch_size-10_num_samples-261_bandLen-21'
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last4,']')
print('# shap files found so far....[',len(shap_files),']')
last4 = len(shap_files)


# # *GOAL:* Determine how long it will take @ 20 tasks [10 indices per subset]/11 nodes [10 CPUs each]/60 minutes
# #### 50 Min Occurrences - __UNLIMITED__ CRON/60 minutes: pdsh across adaptjh2 overnight then switched to ilab202-ilab212 = 20 tasks per node, max of 10 nodes with 10 CPUs each
# #### */60 * * * *  pdsh -w ilab[202,203,204,205,206,207,208,209,210,211,212] sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-aggregate-20trials.sh 

# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240204_20trials_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last2,']')
print('# shap files found so far....[',len(shap_files),']')
last2 = len(shap_files)


# In[ ]:


shap_files[0]


# # *GOAL:* Determine if we can schedule 100 tasks [50 indices per subset]/10 nodes [10 CPUs each]/4 hours
# #### *UPDATED 2/5 - 5 AM:*  00 02,06,10,14,16 05 02 *  pdsh -w ilab[202,203,204,205,206,208,209,210,211,212] sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-aggregate.sh 
# #### *UPDATED 2/5 - 5:30 AM* - linked output from job below to aggregate results across runs (check for duplicates later)
# #### 50 Min Occurrences - **SCHEDULED** CRON: pdsh across ilab202-ilab212 w/ cron [00 02,07,12,17 04 02] = 4 jobs {1 per 4/5 hours} - 100 tasks per node, max of 10 nodes with 10 CPUs each
# #### 1st Scheduled cron: 00 02,07,12,17 04 02 *  pdsh -w ilab[202,203,204,205,206,208,209,210,211,212] sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-aggregate.sh

# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240204_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 
#shap_files = glob.glob(results_dir_cpu + '**/**/PERMUTATION_IMPORTANCE_VALUES/*.shap_values0to50',  recursive = True) 
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))

print('# shap files found on last check ....[',last3,']')
print('# shap files found so far....[',len(shap_files),']')
last3 = len(shap_files)


# # Valid, but old job, before adding filtering per shap file across outputs... results linked into output above 
# #### *UPDATED 2/5 - 6:26 AM* - manually stopped stalled jobs (base) gtamkin@ilab203:/home/gtamkin$ pdsh -w ilab[202,203,204,205,206,207,208,209,210,211,212] ps -ef | grep gtamkin | grep 20240203_collection-ALL | more
# ### 50 Min Occurrences - pdsh across ilab202-ilab212 - Expecting more final shap files than original list because of multiple output dirs - logic before filtering was applied
# ## 50 tasks per node = anecdotal from one node ilab208: Time elapsed on node [ilab208] = (hh:mm:ss.ms) 2:09:05.470882

# In[ ]:


#(ilab-tensorflow) gtamkin@ilab209:/home/gtamkin$ pdsh -w ilab[202,203,204,205,206,208,209,210,211,212] sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-aggregate.sh 2>&1 | tee -a /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-44/ilab202_212.out &

# cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
# /panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
# export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo
# sleep 5
# /app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
#    --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447 \
#    --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/random_sets/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447.pkl \
#    -hf /explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/bands/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif \
#    -t 10
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'
shap_files = glob.glob(results_dir_cpu + '**/**/*.shap_values0to50',  recursive = True) 

print('# shap files found on last check ....[',last4,']')
print('# shap files found so far....[',len(shap_files),']')
last4 = len(shap_files)


# # 50 Min Occurrences - ilab213: OSError: [Errno 12] Cannot allocate memory

# In[ ]:


#100 tasks per node = 
#50 tasks per node = ilab208: Time elapsed on node [ilab208] = (hh:mm:ss.ms) 2:09:05.470882
#
# ilab213:     self._launch(process_obj)
# ilab213:   File "/app/jupyter/ilab/tensorflow-kernel/lib/python3.8/multiprocessing/popen_fork.py", line 70, in _launch
# ilab213:     self.pid = os.fork()
# ilab213: OSError: [Errno 12] Cannot allocate memory


#(ilab-tensorflow) gtamkin@ilab209:/home/gtamkin$ pdsh -w ilab213 sh /home/gtamkin/_AGB-dev/monte-carlo/monte-carlo/tests/pdsh-aggregate.sh 2>&1 | tee -a /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/exp_random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447.out &

# cd /panfs/ccds02/home/gtamkin/dev/AGB/monte-carlo/monte-carlo
# /panfs/ccds02/app/modules/anaconda/platform/x86_64/rhel/8.6/3-2022.05/bin/conda activate ilab-tensorflow
# export PYTHONPATH=/explore/nobackup/people/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion:/explore/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo
# sleep 5
# /app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
#    --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447 \
#    --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mlruns/random_sets/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447.pkl \
#    -hf /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif \
#    -t 5000
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447/806638708288219801'
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.model:' | wc -l")
get_ipython().system("ls -alRt $results_dir_cpu/MODELS/ | grep '\\]\\.test' | wc -l")
get_ipython().system('ls -alRt $results_dir_cpu/PERMUTATION_IMPORTANCE_VALUES/ | grep shap |wc -l')

shap_files = glob.glob(results_dir_cpu + '**/*.shap_values0to50',  
                       recursive = True) 
print(shap_files)


# # Experimenting below......

# In[ ]:


from datetime import datetime
start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'
shap_files = glob.glob(results_dir_cpu + '**/**/PERMUTATION_IMPORTANCE_VALUES/*.shap_values0to50',  recursive = True) 
print('# shap files found so far....[',len(shap_files),']')
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))


# In[ ]:


import glob
import json
import fnmatch
import os
from pathlib import Path
from time import time

results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240203_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'

def find_files_iglob():
    return glob.iglob(results_dir_cpu + '**/**/*.shap_values0to50', recursive=True)


def find_files_oswalk():
    for root, dirnames, filenames in os.walk(results_dir_cpu):
        for filename in fnmatch.filter(filenames, '*.shap_values0to50'):
            yield os.path.join(root, filename)

def find_files_rglob():
    return Path(results_dir_cpu).rglob('*.shap_values0to50')

t0 = time()
for f in find_files_oswalk(): pass    
t1 = time()
for f in find_files_rglob(): pass
t2 = time()
for f in find_files_iglob(): pass 
t3 = time()
print(t1-t0, t2-t1, t3-t2)


# In[ ]:


import fnmatch
import os
from datetime import datetime

def recursive_glob(rootdir='.', pattern='*'):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if fnmatch.fnmatch(filename, pattern)]

start_time = datetime.now()
results_dir_cpu = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/random_sets_20240204_collection-ALL_max_occurrences-50_batch_size-10_num_samples-3227_bandLen-447'
shap_files = recursive_glob(results_dir_cpu, '*.shap_values0to50')
print('# shap files found so far....[',len(shap_files),']')
time_elapsed = datetime.now() - start_time
print('Time elapsed = (hh:mm:ss.ms) {}'.format(time_elapsed))


# In[ ]:


print(shap_files[0])
import re
m = re.search('\[\[(.*?)\]\]', str(shap_files[0]))
#m = re.search(r"(?<=[[).*?(?=]])", str(shap_files[0]))
if m:
    found = m.group(1)
    print('substring = ',found)

currentStr = shap_files[0]
#currentStr = 'wrongemboyo'
if (currentStr.find(found) > 0):
    print('substring [', found, '] exists at index', str(currentStr.find(found)), ' in ', currentStr)
else:
     print('No substring [', found, '] found in ', currentStr)
   


# In[ ]:




