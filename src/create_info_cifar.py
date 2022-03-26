import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data_path = '/home/vladislav/Data/CIFAR100'
    info_path = '/home/vladislav/Data/CIFAR100/info.csv'
    info = {'image_path': [], 'label': [], 'int_label': [], 'part': []}
    class_i, synset_mapping = 0, {}

    all_labels = os.listdir(os.path.join(data_path, 'TRAIN'))
    le = LabelEncoder()
    le.fit(all_labels)

    for label_name in os.listdir(os.path.join(data_path, 'TRAIN')):
        for image_name in os.listdir(os.path.join(data_path, 'TRAIN', label_name)):
            info['image_path'].append(os.path.join('TRAIN', label_name, image_name))
            info['label'].append(label_name)
            info['int_label'].append(le.transform([label_name])[0])
            info['part'].append('train')

    for label_name in os.listdir(os.path.join(data_path, 'TEST')):
        for image_name in os.listdir(os.path.join(data_path, 'TEST', label_name)):
            info['image_path'].append(os.path.join('TEST', label_name, image_name))
            info['label'].append(label_name)
            info['int_label'].append(le.transform([label_name])[0])
            info['part'].append('test')

    pd.DataFrame(info).to_csv(info_path, index=False)
