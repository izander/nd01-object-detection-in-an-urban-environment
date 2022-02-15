import argparse
import glob
import os
import random
import shutil

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    trval_dir = os.path.join(data_dir, 'training_and_validation')    

    if os.path.exists(train_dir) == False:
       os.makedirs(train_dir)
    if os.path.exists(val_dir) == False:
       os.makedirs(val_dir)
    if os.path.exists(test_dir) == False:
       os.makedirs(test_dir)
        
    all_files = [filename for filename in glob.glob(f'{trval_dir}/*.tfrecord')]
    np.random.shuffle(all_files)
    print('all files: ', len(all_files))
    
    train_files, val_files = np.split(all_files, [int(len(all_files)*0.8)])
    
    print('train files: ', len(train_files))
    print('val files: ', len(val_files))
   
    logger.info('Moving data')
    for data in train_files:
        shutil.move(data, train_dir)
    
    for data in val_files:
        shutil.move(data, val_dir)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)