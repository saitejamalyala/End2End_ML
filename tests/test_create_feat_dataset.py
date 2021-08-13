import unittest
import os
from ..src.create_feat_dataset import *
from ..src.create_dataset import *
from .. import config

class TestCreateFeatDataset(unittest.TestCase):
    def test_feat_dataset(self):
        clean_data=load_clean_data()
        create_feat_dataset(clean_data)
        self.assertTrue(os.path.isfile(os.path.join(os.getcwd(),*config['ds_features']) and os.path.isfile(os.path.join(os.getcwd(),*config['ds_target']))))


if __name__ == '__main__':
    unittest.main()