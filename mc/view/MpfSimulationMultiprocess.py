import sys,os
import argparse

from mc.model.simulations.mpf.configureMpfRun import ConfigureMpfRun
from mc.model.simulations.mpf.MpfWorkflow import MpfWorkflow
from datetime import datetime
import socket
import multiprocessing as multiprocessing


###############################################################################################
########################### MPF Simulation ####################################################
###############################################################################################

# -------------------------------------------------------------------------------
# main
#
# ./MpfSimulation.py --config
# /panfs/ccds02/home/gtamkin/dev/AGB/mpf-model-factories/MultiPathFusion/multi_path_fusion/config/exp_all21bands.json
# --bandList
# "1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21"
# -dp
# /explore/nobackup/projects/ilab/data/AGB/test/beta_pmm/
# -hf
# MLBS_2018_541567.6_4136443.0_542567.6_4137443.0/MLBS_2018_hyperspectral_indices.tif
# -tfa
# MLBS_2018_541567.6_4136443.0_542567.6_4137443.0/MLBS_2018_FSDGtif_CHM_warp.tif
# -e
# Allbands
# -o
# /explore/nobackup/projects/ilab/data/AGB/test/mlruns/12212023-all21bands
# -p
# 10
# -t
# 100

def main():
    # Process command-line args. 
    desc = 'This application run an end-to-end MPF simulation'

    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    print("Your Computer Name is:" + hostname)
    print("Your Computer IP Address is:" + IPAddr)

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--config',
                        required = True,
                        type=str,
                        help='config file path for loading settings')

    parser.add_argument('--bandListFile',
                        default='',
                        required = False,
                        type=str,
                        help='path to file containing lists of bands to process (mutually exclusive with --bandList')

    parser.add_argument('--shapArchive',
                        default='',
                        required = False,
                        type=str,
                        help='path to directory containing previously processed shap files')

    parser.add_argument('--bandList',
                        help='list of bands to process (mutually exclusive with --bandListFile', type=str)

    parser.add_argument('-dp',
                        help='root data path')

    parser.add_argument('-hf',
                        help='path to hyperspectral indices file')

    parser.add_argument('-tfa',
                        help='path to truth file A')

    parser.add_argument('-tfb',
                        help='path to truth file B')

    parser.add_argument('-e',
                       help='experiment name')

    parser.add_argument('-o',
                        default='.',
                        help='root path of output directory')

    parser.add_argument('-p',
                        default=10,
                        help='number of concurrent processes to run')

    parser.add_argument('-t',
                        default=10,
                        help='number of batches of band subsets for each concurrent process')

    parser.add_argument(
        "--clean", "--clean", required=False, dest='cleanbool',
        action='store_true', help="Force cleaning of generated artifacts prior to run (e.g, model files)."
    )
    parser.add_argument(
        "--archive", "--archive", required=False, dest='archivebool',
        action='store_true', help="Archive interim artifacts."
    )

    args = parser.parse_args()
    start_time = datetime.now()

    # Run the process.
    mpfWorkflowConfig = (ConfigureMpfRun(args.config, args.bandList, args.bandListFile, args.shapArchive,
                                         args.dp, args.hf, args.tfa, args.tfb, args.e,
                                         args.o, args.cleanbool, args.archivebool, args.p, args.t)).config
    mpfWorkflowConfig.workflow.set_paths(mpfWorkflowConfig.outDir,
                   mpfWorkflowConfig.mlflow_config['EXPERIMENT_ID'])
    try:

        random_sets_r = mpfWorkflowConfig.workflow.get_band_sets(
            args.bandList, args.bandListFile, args.shapArchive, mpfWorkflowConfig)
        num_workers = len(random_sets_r)
        if (num_workers > 0):
            from multiprocessing.pool import Pool
            items = [(random_sets_r[i], mpfWorkflowConfig) for i in range(0, len(random_sets_r))]

            print('max processes: ', multiprocessing.cpu_count(), ' processes requested from pool: ', num_workers)
            print(f'Starting pool.starmap_async() for {str(len(items))} band sets:' + str(random_sets_r), flush=True)
            with Pool(num_workers) as pool:
                # issue tasks to process pool
                result = pool.starmap_async(mpfWorkflowConfig.workflow.process_band_list, items)
                # iterate results
                for result in result.get():
                    print(f'Got result: {result}', flush=True)
        else:
            print('Processing complete.  See:', mpfWorkflowConfig.outDir)

        if (mpfWorkflowConfig.workflow != None):
            mpfWorkflowConfig.workflow.cleanup()
            
    except OSError as err:
        print("OS error:", err)
    except Exception as inst:
        print(type(inst))  # the exception type
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,
        # but may be overridden in exception subclasses

    finally:

        time_elapsed = datetime.now() - start_time
        print('Time elapsed on node [{}] = (hh:mm:ss.ms) {}'.format(hostname, time_elapsed))

# -------------------------------------------------------------------------------
# Invoke the main
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

