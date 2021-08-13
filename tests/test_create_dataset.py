import unittest
from unittest import TestCase
#from End2End_ML.src.create_dataset import *
from ..src.create_dataset import *
import os
from .. import config

class TestCreateDataset(TestCase):

    def test_create_dataset(self):
        print(os.getcwd())
        load_raw_data()
        raw_data_path = os.path.join(os.getcwd(),*config['raw_data_path'])
        self.assertTrue(os.path.isfile(raw_data_path))


if __name__ == '__main__':
    unittest.main()