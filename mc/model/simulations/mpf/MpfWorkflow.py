import json
import os
import time
import keras

from multi_path_fusion.src.utils.mlflow_helpers import setup_mlflow, log_params
from multi_path_fusion.src.utils.data_generator_helpers import load_data_generator
from multi_path_fusion.src.utils.model_helpers import get_model_factory

from multi_path_fusion.src.training.train import train

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

        self.mpfConfig.model_path = os.path.join(self.mpfConfig.modelDir, self.mpfConfig.model_name)
        if (not os.path.exists(self.mpfConfig.model_path)) or (not hasattr(self.mpfConfig, 'model_factory')):

            self.logger.info('\nCreating model: ' + self.mpfConfig.model_path)

            self.mpfConfig.model_type = self.mpfConfig.model_config.get("model_type", "Sequential")
            self.mpfConfig.model_factory = get_model_factory(self.mpfConfig.model_type)
            model = self.mpfConfig.model_factory.create_model(self.mpfConfig.model_config,
                                                              self.mpfConfig.data_generator_config)
            model = self.mpfConfig.model_factory.compile_model(model, self.mpfConfig.model_config)

            model.save(self.mpfConfig.model_path)

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

        return self.mpfConfig

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
#            self.mpfConfig.history2 = \
#                model_factory.train_model(model, train_generator, validate_generator, model_config)

            t1 = time.time()
            total_time = t1 - t0

            mlflow.log_param("Training time", total_time)

            print()
            print("Total time for model.fit() processing is: " + str(total_time) + " seconds.")

            if model_type == "XGBoost":
                model_factory.summary(model)

    def _get_shap_values(self, model, test_generator, X):

        keyword = self.keywordAll
        import shap
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)

        model21 = model
        self.mpfConfig.explainer21 = shap.KernelExplainer(model21.predict, X.iloc[:50, :])
        self.mpfConfig.shap_values21_0to50 = self.mpfConfig.explainer21.shap_values(X.iloc[0:50, :], nsamples=500)
        shap.summary_plot(self.mpfConfig.shap_values21_0to50[0], X.iloc[0:50, :], plot_type="bar", feature_names=self.keywordAll)

    def _get_shap_values2(self, model, test_generator, X):


        keyword = self.keywordAll
        import shap
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)

        self.mpfConfig.exact_explainerAll  = shap.Explainer(model.predict, X)
        self.mpfConfig.exact_explainer.shap_values_explanationAll= self.mpfConfig.exact_explainerAll(X)

        self.mpfConfig.exact_explainer  = shap.Explainer(model.predict, X.iloc[:50, :])
        self.mpfConfig.exact_explainer.shap_values_explanation0to50 = self.mpfConfig.exact_explainer(X.iloc[:50, :])
#        self.mpfConfig.Explainer.shap_values0to50 = self.mpfConfig.Explainer.shap_values(X.iloc[0:50, :], nsamples=500)

        self.mpfConfig.kernel_explainer = shap.KernelExplainer(model.predict, X.iloc[:50, :])
        self.mpfConfig.kernel_explainer.shap_values_explanation0to50 = self.mpfConfig.kernel_explainer(X.iloc[0:50, :])

#        self.mpfConfig.shap_values0to50 = self.mpfConfig.explainer.shap_values(X.iloc[0:50, :], nsamples=500)
        # self.mpfConfig.shap_values51to100 = self.mpfConfig.explainer.shap_values(X.iloc[50:99, :], nsamples=500)
        # self.mpfConfig.shap_value101to150 = self.mpfConfig.explainer.shap_values(X.iloc[100:149, :], nsamples=500)
        # self.mpfConfig.shap_values50 = self.mpfConfig.explainer.shap_values(X.iloc[280:330, :], nsamples=500)

        shap.summary_plot(self.mpfConfig.exact_explainer.shap_values0to50[0], X.iloc[0:50, :], plot_type="bar", feature_names=keyword)
        shap.summary_plot(self.mpfConfig.kernel_explainer.shap_values0to50[0], X.iloc[0:50, :], plot_type="bar", feature_names=keyword)

        shap.summary_plot(shap_values0to50[0], X.iloc[280:330, :], plot_type="bar", feature_names=keyword)
        print(len(self.mpfConfig.shap_values))
        print(self.mpfConfig.shap_values)
        return self.mpfConfig.explainer, self.mpfConfig.shap_values

    def run_trials(self):

        model = self.mpfConfig.model
        model_factory = self.mpfConfig.model_factory
        test_generator = self.mpfConfig.test_generator

        self.mpfConfig.predictions = model_factory.predict(model, test_generator)
        self.mpfConfig.predictions2 = model_factory.predict(model, test_generator)
        self.mpfConfig.test_results = model_factory.evaluate(model, test_generator)

        # debugging
        print(f'predictions: {self.mpfConfig.predictions }')
        print(f'test results: {self.mpfConfig.test_results}')

#        model_factory.log_metrics(test_results)
        #        model_factory.plot_metrics(model, test_generator, history)

        # debugging
        print("Finished predictions successfully")

        import pandas as pd
        df = pd.DataFrame(data=test_generator.file_x_stack)
        dft = df.transpose()
        X = dft

        explainer, shap_values = self._get_shap_values(model, test_generator, X)
        explainer2, shap_values2 = self._get_shap_values(model, test_generator, X)
        print("Finished shap successfully")

    def selector(self):
        pass

    def modeler(self):
        pass
