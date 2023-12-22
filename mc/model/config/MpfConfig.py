from datetime import datetime
import json
import os

# -------------------------------------------------------------------------------
# class MpfConfig
# -------------------------------------------------------------------------------
class MpfConfig(object):
    DATE_FORMAT = '%m-%d-%Y'

    STATES = {'PENDING': 'Pending',
              'RUNNING': 'Running',
              'COMPLETE': 'Complete',
              'FAILED': 'Failed'}

    CONFIG_FILE_KEY = 'configFile'
    END_DATE_KEY = 'endDate'
    EPSG_KEY = 'epsg'
    IN_DIR_KEY = 'inputDirectory'
    LRX_KEY = 'lrx'
    LRY_KEY = 'lry'
    NUM_PROCS_KEY = 'numProcesses'
    NUM_TRIALS_KEY = 'numTrials'
    OUT_DIR_KEY = 'outputDirectory'
    PHASE_KEY = 'phase'
    PRES_FILE_KEY = 'presenceFile'
    SPECIES_KEY = 'species'
    START_DATE_KEY = 'startDate'
    STATE_KEY = 'state'
    TOP_TEN_KEY = 'topTen'
    ULX_KEY = 'ulx'
    ULY_KEY = 'uly'

    DEFAULT_PROCESSES = 10
    DEFAULT_TRIALS = 10
    MAXIMUM_PROCESSES = 2000
    MAXIMUM_TRIALS = 10000

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self):

        self.phase = 'Unknown'
        self.setStatePending()
        self.configFile = None

        self.numProcesses = MpfConfig.DEFAULT_PROCESSES
        self.numTrials = MpfConfig.DEFAULT_TRIALS

        self.startDate = None
        self.endDate = None

        self.inDir = '.'
        self.outDir = '.'

        self.presFile = None
        self.species = 'species'

        self.ulx = None
        self.uly = None
        self.lrx = None
        self.lry = None
        self.epsg = None

        self.topTen = None

        self.configFile = None
        self.bandList  = None
        self.dataPath = None
        self.hyperspectralFilePath = None
        self.truthFileA = None
        self.truthFileB = None
        self.experimentName = None
        self.outDir = None
        self.numProcesses = None
        self.numTrials = None


    # ---------------------------------------------------------------------------
    # fromDict
    # ---------------------------------------------------------------------------
    def fromDict(self, inDict):

        self.configFile = inDict[MpfConfig.CONFIG_FILE_KEY]
        self.setEndDate(inDict[MpfConfig.END_DATE_KEY])
        self.setEPSG(inDict[MpfConfig.EPSG_KEY])
        self.setInDir(inDict[MpfConfig.IN_DIR_KEY])
        self.setLrx(inDict[MpfConfig.LRX_KEY])
        self.setLry(inDict[MpfConfig.LRY_KEY])
        self.setNumProcs(inDict[MpfConfig.NUM_PROCS_KEY])
        self.setNumTrials(inDict[MpfConfig.NUM_TRIALS_KEY])
        self.setOutDir(inDict[MpfConfig.OUT_DIR_KEY])
        self.phase = inDict[MpfConfig.PHASE_KEY]
        self.setPresFile(inDict[MpfConfig.PRES_FILE_KEY])
        self.setSpecies(inDict[MpfConfig.SPECIES_KEY])
        self.setStartDate(inDict[MpfConfig.START_DATE_KEY])
        self.state = inDict[MpfConfig.STATE_KEY]
        self.topTen = inDict[MpfConfig.TOP_TEN_KEY]
        self.setUlx(inDict[MpfConfig.ULX_KEY])
        self.setUly(inDict[MpfConfig.ULY_KEY])

    # ---------------------------------------------------------------------------
    # getConfigFile
    # ---------------------------------------------------------------------------
    def getConfigFile(self):
        return self.configFile

    # ---------------------------------------------------------------------------
    # initializeFromFile
    # ---------------------------------------------------------------------------
    def initializeFromFile(self, inFile):

        with open(inFile, 'r') as f:
            jsonStr = f.read()
            jsonConfig = json.loads(jsonStr)
            self.fromDict(jsonConfig)

    # ---------------------------------------------------------------------------
    # initializeFromValues
    # ---------------------------------------------------------------------------
    def initializeFromValues(self, configFile, bandList, dataPath, hyperspectralFile,
                             truthFileA, truthFileB, experiment, outDir, numProcs, numTrials):

        self.setConfigFile(configFile)
        self.setBandList(bandList)
        if dataPath:
            self.setDataPath(dataPath)
            if hyperspectralFile: self.setHyperspectralFile(dataPath, hyperspectralFile)
            if truthFileA: self.setTruthFileA(dataPath, truthFileA)
            if truthFileB: self.setTruthFileB(dataPath, truthFileB)
        self.setExperimentName(experiment)
        self.setOutDir(outDir)
        self.setNumProcs(numProcs)
        self.setNumTrials(numTrials)

    # ---------------------------------------------------------------------------
    # setConfigFile
    # ---------------------------------------------------------------------------
    def setConfigFile(self, configFile):

        self.configFile = None
        if configFile and (not os.path.exists(configFile)):
            raise RuntimeError('A valid configuration file is required.')
        else:
            self.configFile = configFile

    # ---------------------------------------------------------------------------
    # setBamdList
    # ---------------------------------------------------------------------------
    def setBandList(self, bandList):

        if bandList:
            self.bandList = [int(item) for item in bandList.split(',')]

    # ---------------------------------------------------------------------------
    # setDataPath
    # ---------------------------------------------------------------------------
    def setDataPath(self, dataPath):

       if dataPath and (not os.path.exists(dataPath) or \
                not os.path.isdir(dataPath)):
            raise RuntimeError('A valid input data path directory is required.')
       else:
            self.dataPath = dataPath

    # ---------------------------------------------------------------------------
    # setHyperspectralFile
    # ---------------------------------------------------------------------------
    def setHyperspectralFile(self, dataPath, hyperspectralFile):

       try:
            self.hyperspectralFilePath = os.path.join(dataPath, hyperspectralFile)
       except Exception as inst:
            raise RuntimeError('A valid hyperspectral file is required.')

    # ---------------------------------------------------------------------------
    # setTruthFileA
    # ---------------------------------------------------------------------------
    def setTruthFileA(self, dataPath, truthFileA):

       try:
            self.truthFileA = os.path.join(dataPath, truthFileA)
       except Exception as inst:
            raise RuntimeError('A valid truth file A is required.')

    # ---------------------------------------------------------------------------
    # setTruthFileB
    # ---------------------------------------------------------------------------
    def setTruthFileB(self, dataPath, truthFileB):

        try:
            self.truthFileB = os.path.join(dataPath, truthFileB)
        except Exception as inst:
            raise RuntimeError('A valid truth file AB is required.')

    # ---------------------------------------------------------------------------
    # setExperimentName
    # ---------------------------------------------------------------------------
    def setExperimentName(self, experimentName):

        if (experimentName):
            self.experimentName = experimentName

    # ---------------------------------------------------------------------------
    # setOutDir
    # ---------------------------------------------------------------------------
    def setOutDir(self, outDir):

       if outDir and (not os.path.exists(outDir) or \
                not os.path.isdir(outDir)):
                try:
                    os.mkdir(outDir)
                    self.outDir = outDir
                except Exception as inst:
                    raise RuntimeError('A valid output directory is required.')
       else:
            self.outDir = outDir

    # ---------------------------------------------------------------------------
    # setNumProcs
    # ---------------------------------------------------------------------------
    def setNumProcs(self, numProcs):

        if numProcs > 0 and numProcs < MpfConfig.MAXIMUM_PROCESSES:
            self.numProcesses = numProcs

    # ---------------------------------------------------------------------------
    # setNumTrials
    # ---------------------------------------------------------------------------
    def setNumTrials(self, numTrials):

        if numTrials > 0 and numTrials < MpfConfig.MAXIMUM_TRIALS:
            self.numTrials = numTrials

    # ---------------------------------------------------------------------------
    # setStateComplete
    # ---------------------------------------------------------------------------
    def setStateComplete(self):
        self.state = MpfConfig.STATES['COMPLETE']

    # ---------------------------------------------------------------------------
    # setStateFailed
    # ---------------------------------------------------------------------------
    def setStateFailed(self):
        self.state = MpfConfig.STATES['FAILED']

    # ---------------------------------------------------------------------------
    # setStatePending
    # ---------------------------------------------------------------------------
    def setStatePending(self):
        self.state = MpfConfig.STATES['PENDING']

    # ---------------------------------------------------------------------------
    # setStateRunning
    # ---------------------------------------------------------------------------
    def setStateRunning(self):
        self.state = MpfConfig.STATES['RUNNING']

    # ---------------------------------------------------------------------------
    # toDict
    # ---------------------------------------------------------------------------
    def toDict(self):

        startDate = None
        endDate = None

        if self.startDate != None:
            startDate = self.startDate.strftime(MpfConfig.DATE_FORMAT)

        if self.endDate != None:
            endDate = self.endDate.strftime(MpfConfig.DATE_FORMAT)

        return {MpfConfig.CONFIG_FILE_KEY: self.configFile,
                MpfConfig.END_DATE_KEY: endDate,
                MpfConfig.EPSG_KEY: self.epsg,
                MpfConfig.LRX_KEY: self.lrx,
                MpfConfig.LRY_KEY: self.lry,
                MpfConfig.NUM_PROCS_KEY: self.numProcesses,
                MpfConfig.NUM_TRIALS_KEY: self.numTrials,
                MpfConfig.IN_DIR_KEY: self.inDir,
                MpfConfig.OUT_DIR_KEY: self.outDir,
                MpfConfig.PHASE_KEY: self.phase,
                MpfConfig.PRES_FILE_KEY: self.presFile,
                MpfConfig.SPECIES_KEY: self.species,
                MpfConfig.START_DATE_KEY: startDate,
                MpfConfig.STATE_KEY: self.state,
                MpfConfig.TOP_TEN_KEY: self.topTen,
                MpfConfig.ULX_KEY: self.ulx,
                MpfConfig.ULY_KEY: self.uly}

    # ---------------------------------------------------------------------------
    # __str__
    # ---------------------------------------------------------------------------
    def __str__(self):


        msg = 'Configuration File: ' + str(self.configFile) + '\n' + \
              'Band List:          ' + str(self.bandList) + '\n' + \
              'Data Path:          ' + str(self.dataPath) + '\n' + \
              'Hyperspectral File: ' + str(self.hyperspectralFilePath) + '\n' + \
              'Truth File A:       ' + str(self.truthFileA) + '\n' + \
              'Truth File B:       ' + str(self.truthFileB) + '\n' + \
              'Experiment:         ' + str(self.experimentName) + '\n' + \
              'Output Directory:   ' + str(self.outDir) + '\n' + \
              'Number of Processes:' + str(self.numProcesses) + '\n' + \
              'Number of Trials:   ' + str(self.numTrials)

        return msg

    # ---------------------------------------------------------------------------
    # write
    # ---------------------------------------------------------------------------
    def write(self):

        configFile = os.path.join(self.outDir, 'config.mpf')

        with open(configFile, 'w') as f:
            self.configFile = configFile
            f.write(json.dumps(self.toDict(), indent=0))  # indent pretty prints


