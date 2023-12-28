from datetime import datetime
import json
import os

from mc.model.config.MonteCarloConfig import MonteCarloConfig

# -------------------------------------------------------------------------------
# class MpfConfig
# -------------------------------------------------------------------------------
class MpfConfig(MonteCarloConfig):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self):

        super().__init__()

        self.bandList  = None
        self.dataPath = None
        self.hyperspectralFilePath = None
        self.truthFileA = None
        self.truthFileB = None
        self.experimentName = None
        self.tracking_uri = None
        self.cfg_path = None
        self.model_path = None
        self.model_name = None
        self.explainer_path = None
        self.shap_path = None
        self.evaluation_path = None


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
    # setBandList
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
    # setExperimentName
    # ---------------------------------------------------------------------------
    def setModelName(self, modelName):

        if (modelName):
            self.model_name = modelName

    # ---------------------------------------------------------------------------
    # __str__
    # ---------------------------------------------------------------------------
    def __str__(self):

        msg = 'Input Config File:  ' + str(self.configFile) + '\n' + \
              'Final State:        ' + str(self.cfg_path) + '\n' + \
              'Band List:          ' + str(self.bandList) + '\n' + \
              'Hyperspectral File: ' + str(self.hyperspectralFilePath) + '\n' + \
              'Truth File A:       ' + str(self.truthFileA) + '\n' + \
              'Truth File B:       ' + str(self.truthFileB) + '\n' + \
              'Tracking URI:       ' + str(self.tracking_uri) + '\n' + \
              'Experiment:         ' + str(self.experimentName) + '\n' + \
              'Model Path:         ' + str(self.model_path) + '\n' +  \
              'Explainer_Path:     ' + str(self.explainer_path) + '\n' + \
              'Shap Path:          ' + str(self.shap_path) + '\n' + \
              'Evaluation_path:    ' + str(self.evaluation_path) + '\n' + \
              'Output Directory:   ' + str(self.outDir) + '\n' + \
              'Number of Processes:' + str(self.numProcesses) + '\n' + \
              'Number of Trials:   ' + str(self.numTrials)

        return msg


