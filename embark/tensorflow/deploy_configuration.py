"""Global container and accessor for flags and their values."""

import re
import yaml

class DeployConfiguration(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = {}

    def _parse_config(self):
        if not self.config_file:
            raise Exception('config_file must not be None')
        with open(self.config_file) as f:
            file_str = ""
            for line in f:
                if not re.match('\s*//.*', line):
                    file_str += line
            self.config = yaml.safe_load(str(file_str))

    def get_operation_config(self):
        return self.config['operation']

    def get_logging_config(self):
        return self.config['logging']

    def get_restore_config(self):
        return self.config['restore']

    def get_data_source_config(self, source_name):
        return self.config['data_source'][source_name]

    def get_optimizer_config(self):
        return self.config['optimizer']

    def get_learning_rate_config(self):
        return self.config['learning_rate']

    def get_data_provider_config(self):
        return self.config['data_provider']

    def get_model_config(self):
        return self.config['model']

    def get_tower_batch_size(self):
        return self.get_operation_config()['tower_batch_size']

    def get_num_gpus(self):
        return self.get_operation_config()['num_gpus']