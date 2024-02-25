#!/usr/bin/env python
# coding: utf-8

# # Generate sets of random indices

import numpy as np

from datetime import datetime
import socket

start_time = datetime.now()

#spectrum = "NEON"
#spectrum = "VI"
spectrum = "ALL"

if (spectrum == "ALL"):

    # aggregate number of indices (426 + 21)
    bandLen = 447  # aggregate number of indices
    batch_size = 10
    max_num_occurences = 100

elif (spectrum == "NEON"):

    # Full indices (#426)
    bandLen = 426  # number indices in index file (large file)
    batch_size = 10
    max_num_occurences = 50

else:

    # Partial indices (#21)
    bandLen = 21  # number of bands in original hyperspectral file
    batch_size = 10
    max_num_occurences = 50

MLBS_2018_band_baseline = []
for i in range(bandLen):
    MLBS_2018_band_baseline.append(str(i + 1))
bands = MLBS_2018_band_baseline

from random import sample
random_sets = []
num_samples = 0

max_num_samples = 15000
not_finished = True
bandFoundArr = np.zeros(bandLen).astype(int)
maxed_out_list = []
lastSavedBands = None
shrinkSet = False
dump = True

while not_finished == True:
    random_set = sample(bands, batch_size)
    random_sets.append(random_set)

    # random_set_string = "["
    # for (z in range(0, len(random_set)))
    #     random_set_string = random_set_string + '_' + int(trim(z))
    # random_set_string = random_set_string + ']'

    for i in range(len(random_set)):
        # get band from band list
        bandNum = random_set[i]

        # increment Occurence value in the cell that contains the band number
        bandFoundArr[int(bandNum)-1] = bandFoundArr[int(bandNum)-1] + 1
    
        for index in range(len(bandFoundArr)):
                if (bandFoundArr[index-1] > max_num_occurences):
                    if index not in maxed_out_list:
                        maxed_out_list.append(index)
                    if (shrinkSet == True):
                        print('remove: ', index)
                        bands.remove(index)
                        print('maxed_out_list of bands = ', maxed_out_list)

    num_samples = num_samples+1
    if (num_samples > max_num_samples):
        print(num_samples,' > ', max_num_samples)
        not_finished = False
        dump = False
        break;    
    if (len(maxed_out_list) == bandLen):
        print('Processing complete! len(maxed_out_list):', len(maxed_out_list))
        not_finished = False
        break;

print('final tally: ', bandFoundArr)
print('num random_sets:', len(random_sets))
#print('random_sets:', random_sets)

if (dump == True):
    date = '20240208'
#    version = date + '_shrink_' + str(shrinkSet)
    version = date

    import pickle
    random_set_file = '/explore/nobackup/projects/ilab/data/AGB/test/mcruns/input/random_sets/random_sets_' + version + '_' \
        + 'collection-' + str(spectrum) + '_max_occurrences-' + str(max_num_occurences) + '_batch_size-' + str(batch_size) + '_num_samples-' + str(num_samples) + '_bandLen-' + str(bandLen) + '.pkl'
    # random_set_file = '/explore/nobackup/projects/ilab/data/AGB/test/mlruns/random_sets/random_sets_' + version + '_' \
    #     + 'collection_' + str(spectrum) + '_batch_size=' + str(batch_size) + '_num_samples=' + str(num_samples) + '_bandLen=' + str(bandLen) + '_' + version + '.pkl'
    pickle.dump(random_sets, open(random_set_file , "wb"))
    print('Saved {', str(len(random_sets)), '} random sets of {', batch_size,'} to file: ', random_set_file)

time_elapsed = datetime.now() - start_time
print('Time elapsed on node [{}] = (hh:mm:ss.ms) {}'.format(socket.gethostname(), time_elapsed))

exit();

# In[10]:


import pickle
random_sets2 = [('2','10'), ('9','1','6','11','2')]
random_set_file2 = '/explore/nobackup/projects/ilab/data/AGB/test/mlruns/random_sets_2'
pickle.dump(random_sets2, open(random_set_file2 , "wb"))


# In[5]:


random_sets_r = pickle.load(open(random_set_file, "rb"))   
#print(random_sets_r)


# In[6]:


print(np.array_equal(random_sets, random_sets_r))


# In[7]:


list1 = random_sets_r;
num_sets = 0
while len(list1) > 0:
    num_sets = num_sets + 1
    popped = list1.pop()
    print(popped, 'num_sets = ', num_sets)

print(list1)


# In[16]:


# SuperFastPython.com
# example of parallel starmap_async() with the process pool
from random import random
from time import sleep
from multiprocessing.pool import Pool
 
# task executed in a worker process
def task(identifier, value):
    # report a message
    print(f'Task {identifier} executing with {value}', flush=True)
    # block for a moment
    sleep(value)
    # return the generated value
    return (identifier, value)
 
# protect the entry point
if __name__ == '__main__':
    # create and configure the process pool
    with Pool() as pool:
        # prepare arguments
        items = [(i, random()) for i in range(10)]
        # issues tasks to process pool
        result = pool.starmap_async(task, items)
        # iterate results
        for result in result.get():
            print(f'Got result: {result}', flush=True)
    # process pool is closed automatically


# In[16]:


from multiprocessing.pool import Pool
import multiprocessing as multiprocessing
import os 
# create and configure the process pool
with Pool() as pool:
    print(pool, pool._processes, multiprocessing.cpu_count(), os.cpu_count() )


# In[14]:


# SuperFastPython.com
# example of using starmap() with the process pool
from random import random
from time import sleep
from multiprocessing.pool import Pool
 
# task executed in a worker process
def task(identifier, value):
    # report a message
    print(f'Task {identifier} executing with {value}', flush=True)
    # block for a moment
    sleep(value)
    # return the generated value
    return (identifier, value)
 
# protect the entry point
if __name__ == '__main__':
    # create and configure the process pool
    with Pool() as pool:
        # prepare arguments
        items = [(i, random()) for i in range(8)]
        print(items)
        # execute tasks and process results in order
        for result in pool.starmap(task, items):
            print(f'Got result: {result}', flush=True)
    # process pool is closed automatically


# In[15]:


import multiprocessing
import itertools

def run(args):
    query, cursor = args
    print("running", query, cursor)

queries = ["foo", "bar", "blub"]
cursor = "whatever"
    
with multiprocessing.Pool(processes=10) as pool:
    args = ((args, cursor) for args in itertools.product(queries))
    results = pool.map(run, args)


# In[12]:


import multiprocessing as mp
import resource

def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

mem()

def memoryhog():
    print('...creating list of dicts...')
    n = 10**5
    l = []
    for i in range(n):
        a = 1000*'a'
        b = 1000*'b'
        l.append({ 'a' : a, 'b' : b })
    mem()

proc = mp.Process(target=memoryhog)
proc.start()
proc.join()

mem()


# In[ ]:




