#!/usr/bin/env python
# coding: utf-8

# # DEMO#3 - Re-run with randint and unique indices per random subset...
# Now - Run Three experiments:
# 1). >= 2 identical runs of 21 VI bands with 10 per subset (compare results)
# 2). 426 NEON bands with 50 per subset
# 3). Aggregated (477) bands with 50 per subset
#
# # Other todos:
# 1). Make sure that random # generator is TRULY random each time AND unique to random subset
# 2). Print out contents of trials (e.g., [42, 310, 6, 88,...]) AND absolute values of ranked shap values
# 3). Cleanup old stuff
# # DEMO#2 - Expand Experiments...

# # DEMO#1 - 01/31/2024 - See https://jh-ml.nccs.nasa.gov/jupyterhub-prism/user/gtamkin/lab/workspaces/auto-0/tree/_AGB-dev/mpf-model-factories/MultiPathFusion/multi_path_fusion/notebooks/glenn/gt_mc_get_stats.ipynb

# # Generate Permutation Importance (i.e., rank) from SHAP files

# In[1]:


import shap
import numpy as np
import pickle
import glob
import pandas as pd
import subprocess
import csv
from pathlib import Path


# In[2]:


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

# In[3]:


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


# In[4]:


# initialize list of lists
headers = ['Group','Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MinBands','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
site = 'MLBS_2018_541567.6_4136443.0_542567.6_4137443.0'


# ## Read batch of shap files for experiment, which contains run-time artifacts (e.g., SHAP values)

# In[5]:


root_data_path = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output'


# In[19]:


#headers = ['Group','Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MinBands','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
permutation_results_VI = [
     # ['VI','NoTop10Index', 'Top10Rank', 'DivBy10K','randitU','21', '32', '10', '10','??','ilab213','30','Pool', 
     #  'VI_randint_U_20240229_5','345536919353575236', '## VI - ilab213:  Time elapsed on node [ilab213] = (hh:mm:ss.ms) ??',
     #  'MLBS_2018_hyperspectral_indices.tif','EarlyWarning'],
     ['VI','NoTop10Index', 'Top10Rank', 'DivBy10K','randitU','21', '32', '10', '10','0:26:36.387707','ilab213','30','Pool', 
      'VI_randint_U_20240229_5','345536919353575236', '## VI - ilab213:  Time elapsed on node [ilab213] = (hh:mm:ss.ms) 0:26:36.387707',
      'MLBS_2018_hyperspectral_indices.tif','EarlyWarning'],
     ['VI','NoTop10Index', 'Top10Rank', 'DivBy10K','randitU','21', '32', '10', '10','0:25:12.317','ilab213','30','Pool', 
      'VI_randint_U_20240229_4','673843053471350852', '## VI - ilab213:  Time elapsed on node [ilab213] = (hh:mm:ss.ms) ??',
      'MLBS_2018_hyperspectral_indices.tif','EarlyWarning'],
     ['VI','NoTop10Index', 'Top10Rank', 'DivBy10K','randitU','21', '32', '10', '10','0:26:52.317282','ilab213','30','Pool', 
      'VI_randint_U_20240229_3','548764528868727045', '## VI - ilab213:  Time elapsed on node [ilab213] = (hh:mm:ss.ms) 0:26:52.317282',
      'MLBS_2018_hyperspectral_indices.tif','EarlyWarning'],
     ['VI','NoTop10Index', 'Top10Rank', 'DivBy10K','randitU','21', '32', '10', '10','0:26:56.519002','ilab213','30','Pool', 
      'VI_randint_U_20240229_2','711241757083075094', '## VI - ilab213:  Time elapsed on node [ilab213] = (hh:mm:ss.ms) 0:26:56.519002',
      'MLBS_2018_hyperspectral_indices.tif','EarlyWarning'],
     ['VI','NoTop10Index', 'Top10Rank', 'DivBy10K','randitU','21', '32', '10', '10','0:27:43.264823','ilab213','30','Pool', 
      'VI_randint_U_20240229_1','711241757083075094', '## VI - ilab213:  Time elapsed on node [ilab213] = (hh:mm:ss.ms) 0:27:43.264823',
      'MLBS_2018_hyperspectral_indices.tif','EarlyWarning'],
]


# In[20]:


#headers = ['Group','Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MinBands','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
permutation_results_ALL = [
     ['_ALL','NoTop10Index', 'Top10Rank', 'DivBy10K','randit','447', '1088', '10', '50','??:13:25','ilab213','40','Pool', 
      'random_sets_20240227_shrink_False_num_samples_1088_bandLen_447','784143931214701529', 
      'ALL - min 50 occurrences: ~?? hours',
      'MLBS_2018_541567.6_4136443.0_542567.6_4137443.0-hyperspectral-aggregate.separate-background-scaled.tif ','EarlyWarning'],    
]


# In[21]:


#headers = ['Group','Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MinBands','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
permutation_results_NEON = [
     ['_NEON','NoTop10Index', 'Top10Rank', 'DivBy10K','randit','426', '1079', '10', '10', '07:00.00','gpu004','40','Pool', 
      'randint_sets_20240225_collection-NEON_min_occurrences-10_batch_size-10_num_samples-1080_bandLen-426','188577871478003488', 
      'NEON on gpu004 - min 10 occurrences: ~7hours',
      'MLBS_2018_Reflectance_reflectance_warp_scaled.tif','EarlyWarning'],
]


# In[22]:


#permutation_results = permutation_results_ALL
permutation_results = permutation_results_VI
#permutation_results = permutation_results_NEON
#permutation_results = permutation_results_ALL + permutation_results_NEON + permutation_results_VI
#permutation_results = permutation_results_ALL + permutation_results_NEON + permutation_results_VI

print(len(permutation_results), '\n',permutation_results)


# In[23]:


# Create the pandas DataFrame
df = pd.DataFrame(permutation_results, columns=headers)
df


# # Loop through result files and create the ranking

# In[24]:


#. Loop through result files and create the ranking
for r_index, row in df.iterrows():
    rank_file = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/output/rank/' + df.at[r_index,'ResultDir'] + '.rank' 
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

        print('Summary array before sort:', bandAbsSumArr)
        ascending_indices = np.argsort(bandAbsSumArr)[:10]
        print('Botton ten before after sort:', ascending_indices)
        minIndex = ascending_indices[0]

        descending_indices = np.argsort(bandAbsSumArr)[::-1][:10]
        maxIndex = descending_indices[0]
        print('Top ten before after sort:', descending_indices)

        pickle.dump(bandAbsSumArr, open(rank_file , "wb"))
        print('Saved:', rank_file)
        
    with np.printoptions(precision=5, suppress=True):
#        print('Final:', bandAbsSumArr[:10])
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


# In[25]:


dfTest = df
dfTest


# In[26]:


print('[ID]-[Group]-[Min]-[Band]-[Trials]-[Scaling]-[Top 10 Rank]-[Top 10 Indices]-[Randomizer]')
#headers = ['Group','Top10Index','Top10Rank','Scaling','Randomizer', '#Bands','#Trials','Band/Trial','MinBands','hh:mm:ss.ms','Host(s)','CPUs/Host','Concurrency','ResultDir','ID','Description','HF', 'Callbacks']
for index, row in df.iterrows():
    print('', str(index).rjust(2), '-',row['Group'].rjust(4),'-', row['MinBands'].rjust(3), '-', row['#Bands'].rjust(3),'-', row['#Trials'].rjust(4), '-', row['Scaling'].rjust(8), '-', row['Top10Rank'], '-', row['Top10Index'], '-', row['Randomizer'])


# In[27]:


pd.set_option('max_colwidth', 800)
pd.set_option('display.colheader_justify', 'left')
dfSummary = df[['Group','MinBands','#Bands','#Trials','Randomizer','Scaling','Top10Rank']] 
dfSummary.style.set_properties(**{'text-align': 'left'})
dfSummary


# In[28]:


from pandas import DataFrame
def left_align(df: DataFrame):
    left_aligned_df = df.style.set_properties(**{'text-align': 'left'})
    left_aligned_df = left_aligned_df.set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])]
    )
    return left_aligned_df


# In[29]:


left_align(dfSummary)


# In[ ]:





# In[17]:


dfSort = dfTest.sort_values(["Group","ID"])
dfSort


# In[ ]:





# In[ ]:




