import json
import os
import time
import keras

from multi_path_fusion.src.utils.mlflow_helpers import setup_mlflow, log_params
from multi_path_fusion.src.utils.data_generator_helpers import load_data_generator
from multi_path_fusion.src.utils.model_helpers import get_model_factory

from multi_path_fusion.src.training.train import train

from mc.model.config.MpfConfig import MpfConfig

import pickle
import shap
import numpy as np

class MpfWorkflow(object):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, mpfConfig, logger=None):
        self.mpfConfig = mpfConfig
        self.mpfConfig.model_name = (self.mpfConfig.mlflow_config['EXPERIMENT_NAME']
            + '::'
            + self.mpfConfig.mlflow_config['EXPERIMENT_ID']
            + '.keras')

        # 1    Index Name=ARI
        # 2    Index Name=CAI
        # 3    Index Name=CRI550
        # 4    Index Name=CRI700
        # 5    Index Name=EVI
        # 6    Index Name=EVI2
        # 7    Index Name=fPAR
        # 8    Index Name=LAI
        # 9    Index Name=MCTI
        # 10    Index Name=MSI
        # 11   Index Name=NDII
        # 12    Index Name=NDLI
        # 13    Index Name=NDNI
        # 14    Index Name=NDVI
        # 15    Index Name=NDWI
        # 16    Index Name=NIRv
        # 17    Index Name=PRIn
        # 18    Index Name=PRIw
        # 19    Index Name=SAVI
        # 20    Index Name=WBI
        # 21    Index Name=Albedo

        #        "bands": [18, 14, 9, 3, 20, 8, 10]}]
        keywordAll = ['ARI', 'CAI', 'CRI550', 'CRI700', 'EVI', 'EVI2', 'fPAR', 'LAI', 'MCTI', 'MSI',
                'NDII', 'NDLI', 'NDNI', 'NDVI', 'NDWI', 'NIRv', 'PRIn', 'PRIw', 'SAVI', 'WBI', 'Albedo']

        print(keywordAll)

        keyword7 = list()
        keyword7.append('PRIw')
        keyword7.append('NDVI')
        keyword7.append('MCTI')
        keyword7.append('CRI550')
        keyword7.append('WBI')
        keyword7.append('LAI')
        keyword7.append('MSI')
        #        print(keyword7)

        self.keywordAll = keywordAll
        self.keyword7 = keyword7

        self.logger = logger

    def _generateFileIndexes(self, maxIndex):

        listOfIndexLists = []
        PREDICTORS_PER_TRIAL = 10

        for i in range(1, self.config.numTrials + 1):
            listOfIndexLists.append(random.sample(range(0, maxIndex - 1),
                                                  PREDICTORS_PER_TRIAL))

        return listOfIndexLists

    def _train(self, context, model_config, data_generator_config, mlflow_config):


        # probably not needed - will delete soon
        # run_name = get_unique_run_name(mlflow_config, model_config.get('model_name', None))
        with mlflow.start_run(experiment_id=mlflow_config['EXPERIMENT_ID'], run_name=model_config.get("model_name", None),
                              description=model_config.get("model_description", ""), nested=False) as run:
            model_factory.autolog()
            log_params(model_config)
            RUN_ID = mlflow.active_run().info.run_id
            print(f"STARTING RUN: {RUN_ID}")

            # TODO: figure out best spot to put this function in, or if there needs to be multiple
            # "extras" depending on where in the training flow you want it to occur
            # put any miscellaneous project-specific code here
            model_factory.extras(model,
                                 context.train_generator,
                                 context.validate_generator,
                                 context.test_generator)

            t0 = time.time()

            # TODO: figure out a way to make this if statement more generic
            # problem is keras model summary best done before training, xgboost summary only able to be done after
            if model_type == "Sequential" or "Keras":
                model_factory.summary(model)
            history = model_factory.train_model(model,
                                                context.train_generator,
                                                context.validate_generator,
                                                context.model_config)

            t1 = time.time()
            total_time = t1 - t0

            mlflow.log_param("Training time", total_time)

            print()
            print("Total time for model.fit() processing is: " + str(total_time) + " seconds.")

            if model_type == "XGBoost":
                model_factory.summary(model)

            predictions = model_factory.predict(model, context.test_generator)
            test_results = model_factory.evaluate(model, context.test_generator)

            # debugging
            print(f'predictions: {predictions}')
            print(f'test results: {test_results}')

            model_factory.log_metrics(test_results)
            #        model_factory.plot_metrics(model, test_generator, history)

            # debugging
            print("Finished successfully")


    def _create_model(self):

        model = None
        if not os.path.exists(self.mpfConfig.modelDir):
            os.mkdir(self.mpfConfig.modelDir)

        self.mpfConfig.model_path = os.path.join(self.mpfConfig.modelDir, self.mpfConfig.model_name +
                                                 '[' + str(self.mpfConfig.bandList)[:] + '].model')
        if (not os.path.exists(self.mpfConfig.model_path)) or (not hasattr(self.mpfConfig, 'model_factory')):

            self.logger.info('\nCreating model: ' + self.mpfConfig.model_path)

            self.mpfConfig.model_type = self.mpfConfig.model_config.get("model_type", "Sequential")
            self.mpfConfig.model_factory = get_model_factory(self.mpfConfig.model_type)
            model = self.mpfConfig.model_factory.create_model(self.mpfConfig.model_config,
                                                              self.mpfConfig.data_generator_config)
            model = self.mpfConfig.model_factory.compile_model(model, self.mpfConfig.model_config)

            model.save(self.mpfConfig.model_path)
            self.model_path = self.mpfConfig.model_path

        else:

            self.logger.info('\nLoading model: ' +  self.mpfConfig.model_path)

            model = keras.models.load_model(self.mpfConfig.model_path)

        self.mpfConfig.model = model
        return self.mpfConfig

    def get_data(self):

        self.logger.info('\nPreparing data for EXPERIMENT: ' + self.mpfConfig.model_name)

        # Loop through the models in the config file
        self.mpfConfig.train_generator = load_data_generator(self.mpfConfig.data_generator_config, 'train')
        self.mpfConfig.validate_generator = load_data_generator(self.mpfConfig.data_generator_config, 'validate')
        self.mpfConfig.test_generator = load_data_generator(self.mpfConfig.data_generator_config, 'test')

        if not os.path.exists(self.mpfConfig.trialsDir):
            os.mkdir(self.mpfConfig.trialsDir)

        self.mpfConfig.trial_path = os.path.join(self.mpfConfig.trialsDir, self.mpfConfig.model_name +
                                                 '[' + str(self.mpfConfig.bandList)[:] + '].test_generator.data')
        if (not os.path.exists(self.mpfConfig.trial_path)):

            self.logger.info('\nSaving test data: ' + self.mpfConfig.trial_path)
            # pickle.dump(self.mpfConfig.test_generator,
            #             open(self.mpfConfig.trial_path, "wb"))

        return self.mpfConfig

    def randomize(self):

        bandLen = int(self.mpfConfig.data_generator_config["num_bands"])

        MLBS_2018_Reflectance_reflectance_warp_baseline = []
        for i in range(bandLen):
            MLBS_2018_Reflectance_reflectance_warp_baseline.append(str(i + 1))
        bands = MLBS_2018_Reflectance_reflectance_warp_baseline
        print('bands to sample ', bands)

        bandOccurenceArr = np.zeros(bandLen).astype(int)
        print(bandOccurenceArr)
        bandAbsSumArr = np.zeros(bandLen)
        bandMaxArr = np.zeros(bandLen)
        bandMinArr = np.zeros(bandLen)
        bandAvgArr = np.zeros(bandLen)

        from random import sample
        random_sets = []
        num_samples = 0
        max_num_samples = 1000000
        batch_size = 10
        max_num_occurences = 10
        not_finished = True
        bandFoundArr = np.zeros(bandLen).astype(int)
        my_list = []
        lastSavedBands = None

        while not_finished == True:
            #    print('\nbands: ', bands)
            random_set = sample(bands, batch_size)
            #    print('\nrandom_set: ', random_set)
            random_sets.append(random_set)

            for i in range(len(random_set)):
                # get band from band list
                bandNum = random_set[i]

                # increment Occurence value in the cell that contains the band number
                bandFoundArr[int(bandNum) - 1] = bandFoundArr[int(bandNum) - 1] + 1

                for j in range(len(bandFoundArr)):
                    if (bandFoundArr[j] > max_num_occurences):
                        index = str(j + 1)
                        if index not in my_list:
                            my_list.append(index)
                            print('remove: ', index)
                            bands.remove(index)
                            lastSavedBands = bands
                            print('current bands = ', bands)
                            print('running tally: ', bandFoundArr)

            num_samples = num_samples + 1
            if (num_samples > max_num_samples):
                print(num_samples, ' > ', max_num_samples)
                not_finished = False
                break;
            if (len(bands) < batch_size):
                batch_size = len(bands)
                #        not_finished = False

            if (len(bands) < 1):
                not_finished = False

        print('final tally: ', bandFoundArr)
        print('bands left to process:', bands)
        print('random_sets:', len(random_sets), '\n', random_sets)

        self.mpfConfig.random_sets = random_sets
        self.mpfConfig.random_sets_path = os.path.join(self.mpfConfig.modelDir,
                                                      self.mpfConfig.model_name +
                                                      '[' + str(self.mpfConfig.bandList)[:] + '].randomized_collection')
        self.logger.info('\nSaving randomized collection: ' + self.mpfConfig.random_sets_path)

        pickle.dump(self.mpfConfig.random_sets,
                    open(self.mpfConfig.random_sets_path, "wb"))

        return self.mpfConfig.random_sets

    def prepare_images(self):
        self.mpfConfig = self._create_model()

    def prepare_trials(self):

        import mlflow
        model = self.mpfConfig.model
        model_factory = self.mpfConfig.model_factory
        model_type = self.mpfConfig.model_type
        mlflow_config = self.mpfConfig.mlflow_config
        model_config = self.mpfConfig.model_config
        train_generator = self.mpfConfig.train_generator
        test_generator = self.mpfConfig.test_generator
        validate_generator = self.mpfConfig.validate_generator

        # probably not needed - will delete soon
        # run_name = get_unique_run_name(mlflow_config, model_config.get('model_name', None))
        with mlflow.start_run(experiment_id=mlflow_config['EXPERIMENT_ID'],
                              run_name=mlflow_config['EXPERIMENT_NAME'],
                              description=model_config.get("model_description", ""), nested=False) as run:
            model_factory.autolog()
            log_params(self.mpfConfig.model_config)
            RUN_ID = mlflow.active_run().info.run_id
            print(f"STARTING RUN: {RUN_ID}")

            # TODO: figure out best spot to put this function in, or if there needs to be multiple
            # "extras" depending on where in the training flow you want it to occur
            # put any miscellaneous project-specific code here
            model_factory.extras(model, train_generator,
                                 validate_generator, test_generator)

            t0 = time.time()

            # TODO: figure out a way to make this if statement more generic
            # problem is keras model summary best done before training, xgboost summary only able to be done after
            if model_type == "Sequential" or "Keras":
                model_factory.summary(model)
            self.mpfConfig.history = \
                model_factory.train_model(model, train_generator, validate_generator, model_config)

            t1 = time.time()
            total_time = t1 - t0

            mlflow.log_param("Training time", total_time)

            print()
            print("Total time for model.fit() processing is: " + str(total_time) + " seconds.")

            if model_type == "XGBoost":
                model_factory.summary(model)

#            self.mpfConfig.predictions = model_factory.predict(model, test_generator)
            self.mpfConfig.test_results = model_factory.evaluate(model, test_generator)

            # debugging
#            print(f'predictions: {self.mpfConfig.predictions}')
            print(f'test results: {self.mpfConfig.test_results}')

            #        model_factory.log_metrics(test_results)
            #        model_factory.plot_metrics(model, test_generator, history)

            # debugging
            print("Finished evaluation successfully")

            # save shap values
            if not os.path.exists(self.mpfConfig.modelDir):
                os.mkdir(self.mpfConfig.modelDir)

            self.mpfConfig.evaluation_path = os.path.join(self.mpfConfig.modelDir,
                                                          self.mpfConfig.model_name +
                                                          '[' + str(self.mpfConfig.bandList)[:] + '].test_results')
            if (not os.path.exists(self.mpfConfig.evaluation_path)):

                self.logger.info('\nSaving evaluation test results: ' + self.mpfConfig.evaluation_path)
                pickle.dump(self.mpfConfig.test_results,
                            open(self.mpfConfig.evaluation_path, "wb"))


    def _get_shap_values(self, model, test_generator, X):

        self.mpfConfig.explainer = shap.KernelExplainer(model.predict, X.iloc[:50, :])
        self.mpfConfig.shap_values0to50 = self.mpfConfig.explainer.shap_values(X.iloc[0:50, :], nsamples=500)

        # save shap values
        if not os.path.exists(self.mpfConfig.permutationImportanceDir):
            os.mkdir(self.mpfConfig.permutationImportanceDir)

        self.mpfConfig.shap_path = \
            os.path.join(self.mpfConfig.permutationImportanceDir, self.mpfConfig.model_name +
                         '[' + str(self.mpfConfig.bandList)[:] + '].shap_values0to50')

        self.mpfConfig.explainer_path = \
            os.path.join(self.mpfConfig.permutationImportanceDir, self.mpfConfig.model_name +
                         '[' + str(self.mpfConfig.bandList)[:] + '].explainer_values0to50')

        # self.logger.info('\nSaving explainer: ' + self.mpfConfig.explainer_path)
        # pickle.dump(self.mpfConfig.explainer,
        #             open(self.mpfConfig.explainer_path, "wb"))

        self.logger.info('\nSaving shap values: ' + self.mpfConfig.shap_path)
        pickle.dump(self.mpfConfig.shap_values0to50,
                open(self.mpfConfig.shap_path, "wb"))

        return self.mpfConfig.explainer, self.mpfConfig.shap_values0to50


    def run_trials(self):

        model = self.mpfConfig.model
        model_factory = self.mpfConfig.model_factory
        test_generator = self.mpfConfig.test_generator

        import pandas as pd
        df = pd.DataFrame(data=test_generator.file_x_stack)
        dft = df.transpose()
        self.mpfConfig.X = dft

        explainer, shap_values = self._get_shap_values(model, test_generator, self.mpfConfig.X)
        print("Finished shap successfully")

        return self.mpfConfig.explainer, self.mpfConfig.shap_values0to50

    def selector(self):
        keyword = self.keyword7
        if (len(self.mpfConfig.test_generator.file_x_stack) > 8):
                keyword = self.keywordAll

        # shap.summary_plot(self.mpfConfig.shap_values0to50[0], self.mpfConfig.X.iloc[0:50, :], plot_type="bar", feature_names=keyword)
        # shap.summary_plot(self.mpfConfig.shap_values0to50, self.mpfConfig.X, plot_type="bar", feature_names=keyword)

    def modeler(self):
        pass

    def cleanup(self):

        # show's over, just save the stuff you want

        if (hasattr(self.mpfConfig, 'X')):
            self.mpfConfig.X = None

        if (hasattr(self.mpfConfig, 'shap_values0to50')):
            self.mpfConfig.shap_values0to50 = None

        if (hasattr(self.mpfConfig, 'train_generator')):
            self.mpfConfig.train_generator = None

        if (hasattr(self.mpfConfig, 'validate_generator')):
            self.mpfConfig.validate_generator = None

        if (hasattr(self.mpfConfig, 'test_generator')):
            self.mpfConfig.test_generator = None

        if (hasattr(self.mpfConfig, 'explainer')):
            self.mpfConfig.explainer = None

        if (hasattr(self.mpfConfig, 'history')):
            self.mpfConfig.history = None

        if (hasattr(self.mpfConfig, 'model')):
            self.mpfConfig.model = None

        if (hasattr(self.mpfConfig, 'workflow')):
            self.mpfConfig.workflow = None

        import pickle
        if (self.mpfConfig.cfg_path != None):
            pickle.dump(self.mpfConfig, open(self.mpfConfig.cfg_path, "wb"))
