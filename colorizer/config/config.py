"""
Configure models and datasets/ data loader from .yaml file
"""
import os
import yaml
import logging


FILE_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))
DEFAULT_CFG_PATH = os.path.join(ROOT_PATH, 'config.yaml')
LOGGER = logging.getLogger(__name__)


class Config():
    """
    Read configuration from .yaml file and create Config Singleton class object
    By default config.yaml in the root directory is read
    """
    _instance = None

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config()
        return Config._instance

    @staticmethod
    def clear():
        Config._instance = None

    def dump(self, filepath=None):
        dump_string = yaml.dump(self.conf)
        if filepath is not None:
            with open(filepath, 'w') as f:
                f.write(dump_string)
        return dump_string

    def __init__(self, filepath=DEFAULT_CFG_PATH):
        if Config._instance is not None:
            raise Exception('Config class is singleton')

        LOGGER.info('Loading configuration from file: %s', filepath)
        self.filepath = filepath

        with open(filepath, 'r') as f:
            self.conf = yaml.safe_load(f.read())
        Config._instance = self

    def __str__(self):
        return 'filepath: %s\nconf: %s' % (self.filepath, self.conf)

    def __getitem__(self, key):
        return self.conf[key]

    def __setitem__(self, key, value):
        self.conf[key] = value
