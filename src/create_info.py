import os
import pandas as pd

if __name__ == '__main__':
    data_path = '/home/vladislav/data/ILSVRC2017_CLS-LOC/ILSVRC'
    info_path = '/home/vladislav/data/ILSVRC2017_CLS-LOC/ILSVRC/info.csv'
    info = {'image_id': [], 'label': [], 'protocol_1': []}

    for image_name in os.listdir(data_path+'/Data/CLS-LOC/train'):
        info['image_id'] = 'Data/CLS-LOC/train/' + image_name

