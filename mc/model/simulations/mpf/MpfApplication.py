import logging
import os

# -------------------------------------------------------------------------------
# class MpfApplication
# -------------------------------------------------------------------------------
class MpfApplication(object):
    COMPLETE_FILE = 'complete.state'
    FAILURE_FILE = 'failed.state'
    PENDING_FILE = 'pending.state'
    RUNNING_FILE = 'running.state'

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, mpfConfig, applicationName, logger=None):

        self.config = mpfConfig
        self.config.phase = self.getPhase()
        #self.config.write()

        if not logger:
            logFileName = os.path.join(mpfConfig.outDir, 'mpf.log')
            logging.basicConfig(format='%(message)s', filename=logFileName)
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)

        self.logger = logger

        if not applicationName:
            raise RuntimeError('An application name must be provided.')

        self.applicationName = applicationName
        self.clipReprojDir = os.path.join(mpfConfig.outDir, 'CLIP_REPROJ')
        self.config.modelDir = os.path.join(mpfConfig.outDir, 'MODELS')
        self.config.bandDir = os.path.join(mpfConfig.outDir, 'BANDS')
        self.config.permutationImportanceDir = os.path.join(mpfConfig.outDir, 'PERMUTATION_IMPORTANCE_VALUES')
        self.finishedDir = os.path.join(mpfConfig.outDir, 'FINISHED')
        self.merraDir = os.path.join(mpfConfig.outDir, 'RAW_MERRA')
        self.trialsDir = os.path.join(mpfConfig.outDir, 'TRIALS')

    # ---------------------------------------------------------------------------
    # logHeader
    # ---------------------------------------------------------------------------
    def logHeader(self):

        self.logger.info('--------------------------------------------------')
        self.logger.info(self.applicationName)
        self.logger.info('--------------------------------------------------')

    # ---------------------------------------------------------------------------
    # getPhase
    # ---------------------------------------------------------------------------
    def getPhase(self):
        raise RuntimeError('This method must be overridden by a subclass.')

