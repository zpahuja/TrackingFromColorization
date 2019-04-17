# -*- coding: utf-8 -*-
import os
import pytest
from colorizer.config.config import Config

TEST_CFG_DIR = os.path.dirname(os.path.realpath(__file__))


def get_config_file(parent_dir):
    for filename in parent_dir.listdir():
        print("Getting config file: %s" % filename)
        if filename.ext == '.yaml':
            return filename


def test_singleton():
    conf1 = Config.get_instance()
    conf2 = Config.get_instance()
    assert conf1 == conf2
    Config.clear()


@pytest.mark.filterwarnings("ignore:MarkInfo")
@pytest.mark.datafiles(TEST_CFG_DIR)
def test_singleton2(datafiles):
    filename = get_config_file(datafiles)
    Config(filename)
    with pytest.raises(Exception):
        Config(filename)
    Config.clear()


@pytest.mark.filterwarnings("ignore:MarkInfo")
@pytest.mark.datafiles(TEST_CFG_DIR)
def test_update_and_dump(datafiles):
    filename = get_config_file(datafiles)
    config = Config(filename)
    assert config['foo']['bar'] == 1
    assert config['foo']['baz'] == 2

    config['foo']['bar'] = {'test': 10}
    config['foo']['baz'] = 3
    assert config['foo']['bar']['test'] == 10
    assert config['foo']['baz'] == 3

    yaml_string = config.dump()
    assert yaml_string == 'foo:\n  bar:\n    test: 10\n  baz: 3\n'

    config.dump(filename)
    Config.clear()
    config = Config(filename)

    assert config['foo']['bar']['test'] == 10
    assert config['foo']['baz'] == 3
