from abc import ABC, abstractmethod

class MonteCarloFactory(ABC):

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def prepare_images(self):
        pass

    @abstractmethod
    def prepare_trials(self):
        pass

    @abstractmethod
    def run_trials(self):
        pass

    @abstractmethod
    def selector(self, context):
        pass

    @abstractmethod
    def modeler(self, context):
        pass

