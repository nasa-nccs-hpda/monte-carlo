import sys
import argparse

from mc.model.simulations.mpf.configureMpfRun import ConfigureMpfRun

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

    parser.add_argument('--bandList',
                        help='list of bands to process', type=str)

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

    args = parser.parse_args()

    # Run the process.
    mpfWorkflowConfig = (ConfigureMpfRun(args.config, args.bandList, args.dp, args.hf, args.tfa, args.tfb,
                    args.e, args.o, args.p, args.t)).config
    mpfWorkflow = mpfWorkflowConfig.workflow

    try:

        # Loop through the models in the config file
        for model_config in mpfWorkflowConfig.models_config:

            mpfWorkflowConfig.model_config = model_config
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

# -------------------------------------------------------------------------------
# Invoke the main
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())

