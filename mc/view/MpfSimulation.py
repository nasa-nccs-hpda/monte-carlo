import sys,os
import argparse

from mc.model.simulations.mpf.configureMpfRun import ConfigureMpfRun

import pickle
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
# -------------------------------------------------------------------------------
def main():
    # Process command-line args. 
    desc = 'This application run an end-to-end MPF simulation'

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
                        help='number of trials for selecting top-ten predictors')

    parser.add_argument(
        "--clean", "--clean", required=False, dest='cleanbool',
        action='store_true', help="Force cleaning of generated artifacts prior to run (e.g, model files)."
    )
    parser.add_argument(
        "--archive", "--archive", required=False, dest='archivebool',
        action='store_true', help="Archive interim artifacts."
    )

    args = parser.parse_args()

    mpfWorkflow = None

    # Run the process.
    mpfWorkflowConfig = (ConfigureMpfRun(args.config, args.bandList, args.bandListFile,
                                         args.dp, args.hf, args.tfa, args.tfb, args.e,
                                         args.o, args.cleanbool, args.archivebool, args.p, args.t)).config
    mpfWorkflow = mpfWorkflowConfig.workflow

    random_sets_r = []

    try:
        if ((args.bandListFile != None) and (len(args.bandListFile) > 0)):
            # read random set list from file
            random_set_file = args.bandListFile
            random_sets_r = pickle.load(open(random_set_file, "rb"))
        elif ((args.bandList != None) and (len(args.bandList) > 0) and (str(args.bandList) == 'random')):
            #TODO get random sets
            random_sets_r = mpfWorkflow.randomize()
        else:
            #TODO get band list from config file
            random_sets_r.append(mpfWorkflowConfig.data_generator_config['branch_inputs'][0]["branch_files"][0]["bands"])

        mpfWorkflowConfig.cfg_path = None
        num_sets = 0
        while len(random_sets_r) > 0:
            num_sets = num_sets + 1
            popped = random_sets_r.pop()
            if (len(popped) < 2):
                print('skipping 1-dimensional band list: ', str(popped[0]))
            else:
                if (type(popped[0]) == str):
                    popped = [eval(i) for i in popped]
                mpfWorkflowConfig.data_generator_config['branch_inputs'][0]["branch_files"][0]["bands"] =  popped
                mpfWorkflowConfig.bandList = popped
                mpfWorkflowConfig.data_generator_config['num_bands'] = len(popped)

                # Save the initial configuration object
                if not os.path.exists(mpfWorkflowConfig.cfgDir):
                    os.mkdir(mpfWorkflowConfig.cfgDir)

                mpfWorkflowConfig.cfg_path = \
                    os.path.join(mpfWorkflowConfig.cfgDir, mpfWorkflowConfig.model_name +
                                 '[' + str(mpfWorkflowConfig.bandList)[:] + '].cfg')

                if (not os.path.exists(mpfWorkflowConfig.cfg_path)):
                    mpfWorkflow.logger.info('\nSaving initial configuration: ' + mpfWorkflowConfig.cfg_path)
                    pickle.dump(mpfWorkflowConfig,
                                open(mpfWorkflowConfig.cfg_path, "wb"))

                # Loop through the models in the config file
                for model_config in mpfWorkflowConfig.models_config:

                    mpfWorkflowConfig.model_config = model_config
                    mpfWorkflowConfig.model_config['layers'][0]['units'] = len(popped)

                    mpfWorkflow.prepare_images()
                    mpfWorkflow.get_data()
                    mpfWorkflow.prepare_trials()
                    mpfWorkflow.run_trials()
                    mpfWorkflow.selector()
                    mpfWorkflow.modeler()

    except OSError as err:
        print("OS error:", err)
    except Exception as inst:
        print(type(inst))  # the exception type
        print(inst.args)  # arguments stored in .args
        print(inst)  # __str__ allows args to be printed directly,
        # but may be overridden in exception subclasses

    finally:
        if (mpfWorkflow != None):
            mpfWorkflow.cleanup()

# -------------------------------------------------------------------------------
# Invoke the main
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

