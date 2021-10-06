import os
import pandas as pd

from utils.img_utils import get_input_image_names
from configs.config import params
from data.data_reader import DataReader, DataLoader


def test():
    test_patches_csv_name = 'test_patches_38-cloud.csv'
    params['img_size'] *= 2
    df_test_img = pd.read_csv(os.path.join(TEST_FOLDER, test_patches_csv_name))
    test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER, if_train=False)
    test_data_reader = DataReader(test_img, annotations=None, img_size=params['img_size'], augment=False,
                                  test=True)
    test_dataset = DataLoader(test_data_reader, params['img_size'])(batch_size=params['batch_size'])
    test_dataset.len = len(test_data_reader)


if __name__ == '__main__':
    GLOBAL_PATH = params['train_dataset_dir']
    TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
    TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')
