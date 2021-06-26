import os
import xml.etree.ElementTree as ET

import pandas as pd

if __name__ == '__main__':
    data_path = '/home/vladislav/data/ILSVRC2017_CLS-LOC/ILSVRC'
    synset_mapping_path = '/home/vladislav/data/ILSVRC2017_CLS-LOC/ILSVRC/LOC_synset_mapping.txt'
    info_path = '/home/vladislav/data/ILSVRC2017_CLS-LOC/ILSVRC/info.csv'
    info = {'image_path': [], 'synset': [], 'label': [], 'int_label': [], 'part': []}
    class_i, synset_mapping = 0, {}

    with open(synset_mapping_path, 'r') as f:
        for line in f.readlines():
            synset, label = line.split(' ', maxsplit=1)
            synset_mapping[synset] = (class_i, label.rstrip())
            class_i += 1

    for image_name in os.listdir(os.path.join(data_path, 'Data', 'CLS-LOC', 'val')):
        try:
            tree = ET.parse(os.path.join(data_path, 'Annotations', 'CLS-LOC', 'val', image_name.replace('JPEG', 'xml')))
            for object in tree.getroot().findall('object'):
                name = object.find('name').text
        except FileNotFoundError:
            continue
        info['image_path'].append(os.path.join(data_path, 'Data', 'CLS-LOC', 'val', image_name))
        info['synset'].append(name)
        info['label'].append(synset_mapping[name][1])
        info['int_label'].append(synset_mapping[name][0])
        info['part'].append('val')

    for folder_name in os.listdir(os.path.join(data_path, 'Data', 'CLS-LOC', 'train')):
        for image_name in os.listdir(os.path.join(data_path, 'Data', 'CLS-LOC', 'train', folder_name)):
            info['image_path'].append(os.path.join(data_path, 'Data', 'CLS-LOC', 'train', folder_name, image_name))
            info['synset'].append(folder_name)
            info['label'].append(synset_mapping[folder_name][1])
            info['int_label'].append(synset_mapping[folder_name][0])
            info['part'].append('train')

    pd.DataFrame(info).to_csv(info_path, index=False)
