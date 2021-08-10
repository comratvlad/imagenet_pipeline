import abc
import os

import cv2
import pandas as pd


class DatasetFeature(metaclass=abc.ABCMeta):
    def __init__(self, folder_path: str, info: pd.DataFrame):
        """
        An abstract class that implements the "Feature" interface
        :param folder_path: path to the folder, contains features - images, crops etc
        :param info: pandas.DataFrame with samples info
        """
        self.folder_path = folder_path
        self.info = info  # rgb_image, synset, label, int_label

    @abc.abstractmethod
    def read(self, index):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class RGBImage(DatasetFeature):
    name = 'rgb_image'

    def read(self, index):
        image_path = os.path.join(self.folder_path, self.info.iloc[index]['image_path'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class Synset(DatasetFeature):
    name = 'synset'

    def read(self, index):
        return self.info.iloc[index]['synset']


class Label(DatasetFeature):
    name = 'label'

    def read(self, index):
        return self.info.iloc[index]['label']


class IntLabel(DatasetFeature):
    name = 'int_label'

    def read(self, index):
        return self.info.iloc[index]['int_label']
