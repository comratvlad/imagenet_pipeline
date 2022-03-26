# Imagenet pipeline

---
This repository contains a complete pipeline for training deep learning models. The main frameworks that are used
here are:
* Pytorch Ligtning
* Hydra

You can fork this repository to customize it for your task, for example, for a kaggle competition.

## Docker
```bash
cd imagenet_pipeline/docker
docker build --tag pytorch-research:ubuntu20.cuda11 .

docker run -it --rm --gpus all --shm-size 20Gb
 -v /path/to/data:/home/mainuser/data
 -v /home/to/source:/home/mainuser/src pytorch-research:ubuntu20.cuda11
 /home/mainuser/src/train.py -cp ../data -cn example +device=0 +n_epochs=5
```

## Pipeline architecture
Class `lib.config_parser.ConfigParser` implements the logic of converting a text configuration file (see data/example.yaml) 
into the objects necessary for training the deep learning model. This class inherits from `pl.LightningModule` and 
implements `train_dataloader`, `val_dataloader` and `configure_optimizers` methods.
The class `lib.base_classifier.BaseClassifierPipeline`, in turn, is inherited from the `lib.config_parser.ConfigParser`
and implements the remaining necessary methods of the `pl.LightningModule` methods.
### Data reading
The data for training is read through the entity "Feature". To implement a feature, you need to inherit your class
from an abstract class `lib.data.features.DatasetFeature`. After specifying such a class in the configuration file,
you will automatically receive the object returned by the `YourFeatureClass.read` method of your class in the batch, and it will be
available by the key `YourFeatureClass.name`.

For example, if we have the following Synset class: 

```python
class Synset(DatasetFeature):
    name = 'synset'

    def read(self, index):
        return self.info.iloc[index]['synset']
```
and specify it in the configuration file like this:
```
sampled_features:
  - lib.data.features.Synset
```
then we will get the following batch:
```python
{'sunset': ['n03764736', 'n03243711', 'n03267436', 'n08545432'] }
```
### Data processing
Before getting into the butch, data can be processed using special functions. These functions must accept features by
their name and have the **kwargs construct in the signature. For example:
```python
def channel_last(rgb_image, **kwargs):
    features_dict = {'rgb_image': np.moveaxis(np.array(rgb_frame), -1, 0)}
    features_dict.update(kwargs)
    return features_dict
```
It is important that the function should return a dictionary that contains all the fields necessary for training. 
In order to use a specific processing function in the pipeline, it must be specified in the configuration file,
for example:
```
# Train ==============================================================================
...
transforms: lib.data.transforms.channel_last
...
# ======================================================================================

# Develop ==============================================================================
...
dev_transforms: lib.data.transforms.identity_mapping
...
# ======================================================================================
```
