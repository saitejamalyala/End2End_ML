import unittest
from unittest import TestCase
#from End2End_ML.src.create_dataset import *
from ..src.create_dataset import *
import os
from .. import config

class TestCreateDataset(TestCase):

    def test_create_dataset(self):
        print(os.getcwd())
        df=load_raw_data()
        raw_data_path = os.path.join(os.getcwd(),*config['raw_data_path'])
        self.assertTrue(os.path.isfile(raw_data_path))

    def test_load_clean_dataset(self):
        df=load_raw_data()
        clean_Data(df)
        clean_data_path = os.path.join(os.getcwd(),*config['clean_data_path'])
        self.assertTrue(os.path.isfile(clean_data_path))

    

if __name__ == '__main__':
    unittest.main()