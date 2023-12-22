#!/usr/bin/python

from mc.model.monte_carlo_factories.MpfMonteCarloFactory import MpfMonteCarloFactory
from MpfApplication import MpfApplication
from MpfConfig import MpfConfig


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

    # ---------------------------------------------------------------------------
    # getPhase
    # ---------------------------------------------------------------------------
    def getPhase(self):
        return 'configure'