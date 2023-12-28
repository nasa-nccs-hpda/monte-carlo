#!/usr/bin/python

from mc.model.monte_carlo_factories.MpfMonteCarloFactory import MpfMonteCarloFactory
from mc.model.simulations.mpf.MpfApplication import MpfApplication
from mc.model.config.MpfConfig import MpfConfig


# -------------------------------------------------------------------------------
# class ConfigureMpfRun
# -------------------------------------------------------------------------------
class ConfigureMpfRun(MpfApplication):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, configFile, bandList, dataPath, hyperspectralFile, truthFileA, truthFileB,
                 experiment, outDir, numProcs=10, numTrials=10, logger=None):

        # Create the MpfConfig object.
        mpfConfig = MpfConfig()

        mpfConfig.initializeFromValues(configFile, bandList, dataPath, hyperspectralFile,
                             truthFileA, truthFileB, experiment, outDir, int(numProcs), int(numTrials))

        super(ConfigureMpfRun, self).__init__(mpfConfig,
                                              'ConfigureMpfRun',
                                              logger)

        # Log what we have so far.
        self.logHeader()
        self.logger.info(str(self.config))

        MpfMonteCarloFactory(mpfConfig, self.logger)

        # Save the initial configuration object
        import os, pickle
        if not os.path.exists(mpfConfig.cfgDir):
            os.mkdir(mpfConfig.cfgDir)

        mpfConfig.cfg_path = \
            os.path.join(mpfConfig.cfgDir, mpfConfig.model_name + '.cfg')
        if (not os.path.exists(mpfConfig.cfg_path)):

            self.logger.info('\nSaving initial configuration: ' + mpfConfig.cfg_path)
            pickle.dump(mpfConfig,
                        open(mpfConfig.cfg_path, "wb"))

    # ---------------------------------------------------------------------------
    # getPhase
    # ---------------------------------------------------------------------------
    def getPhase(self):
        return 'configure'