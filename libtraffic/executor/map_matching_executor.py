from libtraffic.executor.abstract_executor import AbstractExecutor
from logging import getLogger
from libtraffic.utils import get_evaluator


class MapMatchingExecutor(AbstractExecutor):

    def __init__(self, config, model):
        self.model = model
        self.config = config
        self.evaluator = get_evaluator(config)
        self.evaluate_res_dir = './libtraffic/cache/evaluate_cache'
        self._logger = getLogger()

    def train(self, train_dataloader, eval_dataloader):
        assert True  # do nothing

    def evaluate(self, test_data):
        """
        use model to test data

        Args:
            test_data
        """
        result = self.model.run(test_data)
        batch = {'route': test_data['route'], 'result': result, 'rd_nwk': test_data['rd_nwk']}
        self.evaluator.collect(batch)
        self.evaluator.save_result(self.evaluate_res_dir)

    def load_model(self, cache_name):
        assert True  # do nothing

    def save_model(self, cache_name):
        assert True  # do nothing
