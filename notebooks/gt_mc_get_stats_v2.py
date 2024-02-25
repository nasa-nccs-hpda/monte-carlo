#!/usr/bin/env python
# coding: utf-8

# # DEMO#1 - 01/31/2024 - See https://jh-ml.nccs.nasa.gov/jupyterhub-prism/user/gtamkin/lab/workspaces/auto-0/tree/_AGB-dev/mpf-model-factories/MultiPathFusion/multi_path_fusion/notebooks/glenn/gt_mc_get_stats.ipynb

# # DEMO#2 - Expand Experiments...
# Now - Run Three experiments:
# 1). 21 VI bands with 10 per subset
# 2). 426 NEON bands with 50 per subset
# 3). Aggregated (477) bands with 50 per subset
#
# # Other todos:
# 0). Never use minmax() again.....
# 1). Make sure that random # generator is TRULY random each time
# 2). Print out contents of trials (e.g., [42, 310, 6, 88,...])
# 3). Cleanup old stuff
# # # Generate Permutation Importance (i.e., rank) from SHAP files

# In[16]:


import shap
import numpy as np
import pickle
import glob
import pandas as pd
import subprocess
import csv
from pathlib import Path


# In[17]:


def format_band_set(subset):
    indexListIntStr = ''
    count = 0
    for band in subset:
        if (count == 0):
            indexListIntStr = indexListIntStr + str(band)
        else:
            indexListIntStr = indexListIntStr + ', ' + str(band)
        count = count + 1
    return indexListIntStr


# # Read from CSV

# In[18]:


hyperspectral_aggregate_bandnames = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/bands/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate-bandnames.csv'
only_row = None
with open(hyperspectral_aggregate_bandnames) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        only_row = row
        print(row[0])
        print(row[446])
#print(only_row)


# In[19]:


# initialize list of lists
headers = ['Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MinBands','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
site = 'MLBS_2018_541567.6_4136443.0_542567.6_4137443.0'


# ## Read batch of shap files for experiment, which contains run-time artifacts (e.g., SHAP values)

# In[20]:


root_data_path = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output'


# In[21]:


#headers = ['Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Bands/Trial','MinOccurences','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
permutation_results_NEON = [
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','randit','426', '5666', '10', '100', '30:51:00.00','gpu004','40','Pool', 
      'randint_sets_20240215_collection-NEON_max_occurrences-100_batch_size-10_num_samples-5666_bandLen-426','188577871478003488', 
      'NEON on gpu004 - min 100 occurrences: ~31 hours',
      'MLBS_2018_Reflectance_reflectance_warp_scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','randit','426', '5876', '10', '100', '48:56:00.00','gpu001','40','Pool', 
      'randint_sets_20240211_collection-NEON_max_occurrences-100_batch_size-10_num_samples-5876_bandLen-426','286420834988853426', 
      'QQ NEON on ilab202-212 - min 100 occurrences: ~49 hours',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','426', '3153', '10', '50', '52:49:00.00','ilab202-212','10','Pool', 
      'random_sets_20240205_collection-NEON_max_occurrences-50_batch_size-10_num_samples-3153_bandLen-426','980360802055833747', 
      'NEON on ilab213 - min 50 occurrences: ~53 hours',
      'MLBS_2018_Reflectance_reflectance_warp_scaled.tif','EarlyWarning'],
]


# In[22]:


#headers = ['Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Bands/Trial','MinOccurences','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
permutation_results_ALL = [
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','randit','447', '5737', '10', '100','96:13:25','ilab213','40','Pool', 
      'randint_sets_20240211_collection-ALL_max_occurrences-100_batch_size-10_num_samples-5737_bandLen-447','218557492334736085', 
      'ALL on ilab213 - min 100 occurrences: ~96 hours',
      'MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif ','EarlyWarning'],
]


# In[23]:


#headers = ['Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Bands/Trial','MinOccurences','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
permutation_results_VI = [
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','randit','21', '261', '10', '100','26:00:00.00','ilab213','40','Pool', 
      'ilab213-finale-VI','681206886322137043', 'VI on ilab213 - min 100 occurrences: ~26 hours',
      'MLBS_2018_Reflectance_reflectance_warp_scaled.tif','EarlyWarning'],
]


# In[24]:


#permutation_results = permutation_results_ALL
#permutation_results = permutation_results_VI
#permutation_results = permutation_results_NEON
permutation_results = permutation_results_ALL + permutation_results_VI + permutation_results_NEON

print(len(permutation_results), '\n',permutation_results)


# In[25]:


# Create the pandas DataFrame
df = pd.DataFrame(permutation_results, columns=headers)
df


# # Loop through result files and create the ranking

# In[ ]:


#. Loop through result files and create the ranking
for r_index, row in df.iterrows():
    rank_file = root_data_path + '/rank/' + df.at[r_index,'ResultDir'] + '.rank' 
    my_file = Path(rank_file)
    if my_file.is_file():
        bandAbsSumArr = pickle.load(open(rank_file, "rb"))
        print('\n ======== index[', r_index, '] ==========')
        print('Loaded:', rank_file)
    else:
        shap_dir_prefix = root_data_path + '/' + df.at[r_index,'ResultDir'] + '/' + df.at[r_index,'ID'] 
        shap_dir = shap_dir_prefix + '/PERMUTATION_IMPORTANCE_VALUES/'
        print('\n ======== index[', r_index, '] ==========')
        print(r_index,'index: ', shap_dir)

        # Returns a list of names in list files. 
        shap_files = glob.glob(shap_dir + '**/*.shap_values0to50',  
                           recursive = True) 

        # set to max number of hyperspectral indices in experiment
        bandLen = int(df.at[r_index,'#Bands'])
        bandOccurenceArr = np.zeros(bandLen).astype(int)
        bandAbsSumArr = np.zeros(bandLen)

        first_time = True
        index = 0
        # Reload config
        start = left = '[['
        end = right = ']]'

        for shap_vals_path in shap_files:

            index = index+1
            num_shap_files = len(shap_files)
            bandListStr = shap_vals_path[shap_vals_path.index(left)+len(left):shap_vals_path.index(right)]
            bandList = [e for e in bandListStr.split(',')]
            for bandNum in bandList:
                bandOccurenceArr[int(bandNum)-1] = (bandOccurenceArr[int(bandNum)-1]) + 1

            # Reload shap values
            shap_values0to50 = pickle.load(open(shap_vals_path, "rb"))
            num_bins = len(shap_values0to50)
            num_rows = len(shap_values0to50[0])
            num_bands = num_cols = len(shap_values0to50[0][0])

            #  Walk across bins [0,4]
            for _bin in range(0, num_bins):

                # get absolute value of all values in matrix
                shap_values0to50_bin_abs = np.abs(shap_values0to50[_bin])

                # convert numpy matrix to dataframe and print sum of absolute values in each column
                df_shap_values0to50_bin_abs = pd.DataFrame(data=shap_values0to50_bin_abs)

                # create a row (numpy array) where each column is the sum of the absolute values in that vertical column
                shap_values_abs_sum_axis0 = np.sum(df_shap_values0to50_bin_abs, axis=0)

                # sort the row from low to high to determine the rank of the columns (e.g., features)
                shap_values_abs_sum_argsort = np.argsort(shap_values_abs_sum_axis0)

                # loop thru each column and update the corresponding statistics in the global row
                for _col in range(0, num_cols):
                    bandNum = int(bandList[_col])

                    # add the absolute value of the current shape_value to the existing value in the cell that is indexed by the band number
                    # this allows us to keep a running total shap values per feature for downselection later....
                    bandAbsSumArr[bandNum-1] = (bandAbsSumArr[bandNum-1] + df_shap_values0to50_bin_abs[_col].sum())

        ascending_indices = np.argsort(bandAbsSumArr)[:10]
        minIndex = ascending_indices[0]

        descending_indices = np.argsort(bandAbsSumArr)[::-1][:10]
        maxIndex = descending_indices[0]

        pickle.dump(bandAbsSumArr, open(rank_file , "wb"))
        print('Saved:', rank_file)
        
    with np.printoptions(precision=5, suppress=True):
        print('Final:', bandAbsSumArr[:10])
        print('Final.min:', bandAbsSumArr.min(), 'Final.max:', bandAbsSumArr.max())
        print('Final L->H:', np.argsort(bandAbsSumArr)[:10])
        row['Top10Index'] = np.argsort(bandAbsSumArr)[::-1][:10]

        indexList = np.argsort(bandAbsSumArr)[::-1][:10]
        bandList = []
        for index in range(0, 10):
        #for (index < 10):
            bandList.append(str(only_row[indexList[index]]))
            print(indexList[index], only_row[indexList[index]])        
            index = index + 1
        row['Top10Rank'] = bandList

        print('Final H->L:', row['Top10Index'] )
        print('Final max:', row['Top10Index'][0])


# In[ ]:


df


# #df

# In[14]:


print('Summary: [Job Index]-[Top 10 Rank]-[Top 10 Indices]-[Scaling]')
#headers = ['Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MaxEpochs','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
for index, row in df.iterrows():
    print(index, '-',row['Top10Rank'], '-', row['Top10Index'], '-', row['Scaling'])


# In[ ]:





# ## v1 data below...

# In[ ]:


# /app/jupyter/ilab/tensorflow-kernel/bin/python3.8 /panfs/ccds02/nobackup/people/gtamkin/dev/AGB/monte-carlo/monte-carlo/mc/view/MpfSimulationMultiprocess.py \
#    --config ./tests/exp_aggregate_hyperspectral_callbacks_noMinMax.json -o /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/20241020-gpu001-callbacks-divideby10k_noMinMax \
#    --bandListFile /explore/nobackup/projects/ilab/data/AGB/test/mlruns/random_sets/random_sets_20240111_shrink_False_num_samples_1088_bandLen_447 \
#    -hf /explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif \
#    -t 1200 
root_data_path = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/aggregate'

#/explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/202410130-gpu004-40cpus-callbacks-divideby10k_noMinMax/225919640119481430/PERMUTATION_IMPORTANCE_VALUES/
root_data_path = '/explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate'
permutation_results = [
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1088', '10', '50', '10:13:32.980461','gpu001','40','Pool', '202410130-gpu004-40cpus-callbacks-divideby10k_noMinMax','225919640119481430', 
      '### gpu004: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized ("scale_data_method": "None")',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1089', '10', '50', '9:44:57.880395','gpu001','40','Pool', '202410125-gpu001-callbacks-divideby10k_noMinMax','393911649901623133', 
      '### gpu001: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized ("scale_data_method": "None")',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1088', '10', '50', '9:57:19.194150','gpu001','40','Pool', '202410121b-gpu001-callbacks-divideby10k_noMinMax','971589025605583160', 
      '### gpu001: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized ("scale_data_method": "None")',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1088', '10', '50', '9:38:07.808645','gpu001','40','Pool', '202410121-gpu001-callbacks-divideby10k_noMinMax','916501809805875784', 
      '### gpu001: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized ("scale_data_method": "None")',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1088', '10', '50', '9:29:50.877797','gpu001','40','Pool', '202410120-gpu001-callbacks-divideby10k_noMinMax','871912588490665214', 
      '### gpu001: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized ("scale_data_method": "None")',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
     ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1088', '10', '50', 'NoMinutes','ilab[201-212,211]','10','pdsh', '202410120-callbacks-divideby10k_noMinMax','648862663793551468', 
      '### CPU pdsh[201-212,211]: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
    ['NoTop10Index', 'Top10Rank', 'DivBy10K','sample','447', '1088', '10', '50', '4:24:16.993903','ilab[201-212]','10','pdsh', '20241019-callbacks-divideby10k_noMinMax','964586160942144948', 
      '### CPU pdsh[201-212]: 50 Epochs (w/ EarlyStopping callbacks) after scaling NEON by 10,000 - NO bands normalized',
      'hyperspectral-aggregate.separate-background-scaled.tif','EarlyWarning'],
    ['NoTop10Index', 'Top10Rank','minmax','sample','447', '1088', '10', '50', '9:36:55.985009','gpu003','40','Pool', '20240118-callbacks-gpu','504373152637067057', 
      'gpu003: Epochs = 50 (w/ Early callbacks) and explainer.shap_values(X.iloc[0:50, :], nsamples=150) - All bands normalized ("scale_data_method": "minmax")',
      'hyperspectral-aggregate.separate-background.tif','EarlyWarning'],
    ['NoTop10Index', 'Top10Rank','minmax','sample','447', '1088', '10', '50', '9:34:11.607509','gpu003','40','Pool', '20240117-callbacks-gpu','660588493840123655', 
      'gpu003: Epochs = 50 (w/ Early callbacks) and explainer.shap_values(X.iloc[0:50, :], nsamples=150) - All bands normalized ("scale_data_method": "minmax")',
      'hyperspectral-aggregate.separate-background.tif','EarlyWarning'],
    ['NoTop10Index', 'Top10Rank', 'minmax','sample','447', '1088', '10', '3', '9:34:11.607509','gpu003','40','Pool', '20240113_gpu','342218863809278249', 
      'GPU: Epochs = 3 and explainer.shap_values(X.iloc[0:50, :], nsamples=150) - All bands normalized ("scale_data_method": "minmax")',
      'hyperspectral-aggregate.separate-background.tif','EarlyWarning'],
    ['NoTop10Index', 'Top10Rank','minmax','sample','447', '1088', '10', '2', '9:34:11.607509','gpu003','40','Pool', '20240112','789617681324607598', 
      'CPU & GPU: Epochs = 2 and explainer.shap_values(X.iloc[0:50, :], nsamples=500) - All bands normalized ("scale_data_method": "minmax")',
      'hyperspectral-aggregate.separate-background.tif','EarlyWarning'],
]


# In[ ]:




