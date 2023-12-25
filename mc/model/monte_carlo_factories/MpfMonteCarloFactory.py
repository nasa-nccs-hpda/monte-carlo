import json

from mc.model.monte_carlo_factories.MonteCarloFactory import MonteCarloFactory
from multi_path_fusion.src.utils.mlflow_helpers import setup_mlflow

class MpfMonteCarloFactory(MonteCarloFactory):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, mpfConfig, logger=None):

#        self.mpfConfig = mpfConfig
        mpfConfig.mpfWorkflow = None

        with open(mpfConfig.configFile, 'r') as f:
            parms = json.load(f)
            mpfConfig.models_config = parms["models"]
            mpfConfig.data_generator_config = parms["data_generator"]
            mpfConfig.mlflow_config = parms["mlflow"]

        # Override config file values
        if (mpfConfig.outDir != None) and (mpfConfig.outDir != '.'):
            mpfConfig.mlflow_config['TRACKING_URI'] = mpfConfig.outDir
        if (mpfConfig.experimentName != None):
            mpfConfig.mlflow_config['EXPERIMENT_NAME'] = mpfConfig.experimentName

        if (mpfConfig.hyperspectralFilePath):
            mpfConfig.data_generator_config['branch_inputs'][0]['branch_files'][0]['mlbs_year_filepath'] = \
                mpfConfig.hyperspectralFilePath

        if (mpfConfig.bandList):
            mpfConfig.data_generator_config['branch_inputs'][0]['branch_files'][0]['bands'] = \
                mpfConfig.bandList
        #TODO consider overriding band_nums in data_generator_config

        if (mpfConfig.truthFileA):
                mpfConfig.data_generator_config['truth_file_a'] = \
                    mpfConfig.truthFileA
        #TODO consider overriding truth_file_a in data_generator_config

        if (mpfConfig.truthFileB):
                mpfConfig.data_generator_config['truth_file_b'] = \
                    mpfConfig.truthFileB
        #TODO consider overriding truth_file_b in data_generator_config

        mpfConfig.mlflow_config = setup_mlflow(mpfConfig.mlflow_config)

        mpfConfig.workflow = self.init_workflow(mpfConfig, logger)
#        return mpfConfig

    def init_workflow(self, mpfConfig, logger):
        # Lazy initialization of workflow to avoid heavy Tensorflow load until last minute
        from mc.model.simulations.mpf.MpfWorkflow import MpfWorkflow
        return MpfWorkflow(mpfConfig,logger)

    def get_data(self):
        if (self.mpfWorkflow == None): self.init_workflow()
        self.mpfWorkflow.get_data()

    def prepare_images(self):
        if (self.mpfWorkflow == None): self.init_workflow()
        self.mpfWorkflow.prepare_images()

    def prepare_trials(self):
        if (self.mpfWorkflow == None): self.mpfWorkflow = self.init_workflow()
        pass

    def run_trials(self):
        if (self.mpfWorkflow == None): self.mpfWorkflow = self.init_workflow()
        pass

    def selector(self):
        if (self.mpfWorkflow == None): self.mpfWorkflow = self.init_workflow()
        pass

    def modeler(self):
        if (self.mpfWorkflow == None): self.mpfWorkflow = self.init_workflow()
        pass
