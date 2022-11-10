# Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model
This repository contains the reference code for computing SAM (Saliency Attentive Model) saliency maps based on the following paper:

_Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, Rita Cucchiara_  
_Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model_  
_IEEE Transactions on Image Processing, 2018_

Please cite with the following Bibtex code:

```
@article{cornia2018predicting,
  author = {Cornia, Marcella and Baraldi, Lorenzo and Serra, Giuseppe and Cucchiara, Rita},
  title = {{Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model}},
  journal = {IEEE Transactions on Image Processing},
  volume={27},
  number={10},
  pages={5142--5154},
  year = {2018}
}
```
 
The PDF of the article is available at this [link](http://aimagelab.ing.unimore.it/imagelab/pubblicazioni/2018-tip.pdf).

Additional experimental results are reported in the following short paper:

_Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, Rita Cucchiara_  
_SAM: Pushing the Limits of Saliency Prediction Models_  
_Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition Workshops, 2018_

Please cite with the following Bibtex code:

```
@inproceedings{cornia2018sam,
  author = {Cornia, Marcella and Baraldi, Lorenzo and Serra, Giuseppe and Cucchiara, Rita},
  title = {{SAM: Pushing the Limits of Saliency Prediction Models}},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2018}
}
```


## Abstract

Data-driven saliency has recently gained a lot of attention thanks to the use of Convolutional Neural Networks for predicting gaze fixations. In this paper we go beyond standard approaches to saliency prediction, in which gaze maps are computed with a feed-forward network, and we present a novel model which can predict accurate saliency maps by incorporating neural attentive mechanisms. The core of our solution is a Convolutional LSTM that focuses on the most salient regions of the input image to iteratively refine the predicted saliency map. Additionally, to tackle the center bias present in human eye fixations, our model can learn a set of prior maps generated with Gaussian functions. We show, through an extensive evaluation, that the proposed architecture overcomes the current state of the art on two public saliency prediction datasets. We further study the contribution of each key components to demonstrate their robustness on different scenarios.

![image](https://user-images.githubusercontent.com/34588815/196414378-34a16d32-9ac0-4f98-a287-18e4456e8d26.png)

## Requirements
* Python 3.9
* OpenCV 4.6.0
* tensorflow/keras 2.9.0

## Setting up the environment
### Keras (WINDOWS)
1. Go to `%USERPROFILE%` directory on Windows and create a folder `.keras`.
2. Inside it, create `keras.json` with the following content -
### Keras (LINUX)
```
cd ~/.keras
echo '{"floatx": "float32",\
"epsilon": 1e-07,\
"backend": "tensorflow",\
"image_data_format": "channels_first"}' > keras.json
```
## Usage
```python
from sam import SalMap
salmap = SalMap()
salmap.compile()
salmap.load_weights(weights_path="/weights/sam-resnet_salicon_weights.pkl")
salmap.test(samples_dir="/samples")
```

## Pretrained Models
Download one of the following pretrained models and save it in the code folder:
* SAM-VGG trained on SALICON (2015 release): **[sam-vgg_salicon_weights.pkl](https://github.com/marcellacornia/sam/releases/download/1.0/sam-vgg_salicon_weights.pkl)**
* SAM-ResNet trained on SALICON (2015 release): **[sam-resnet_salicon_weights.pkl](https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_salicon_weights.pkl)**
* SAM-ResNet trained on SALICON (2017 release): **[sam-resnet_salicon2017_weights.pkl](https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_salicon2017_weights.pkl)**

## Precomputed Saliency Maps
We provide saliency maps predicted by SAM-VGG and SAM-ResNet for three standard datasets (SALICON, MIT1003 and CAT2000):
* **[SAM-VGG predictions](https://github.com/marcellacornia/sam/releases/download/1.0/sam-vgg_predictions.zip)**
* **[SAM-ResNet predictions](https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_predictions.zip)**

In addition, we provide saliency maps predicted by SAM-ResNet on the new release of the SALICON dataset:
* **[SAM-ResNet predictions (SALICON 2017)](https://github.com/marcellacornia/sam/releases/download/1.0/sam-resnet_predictions_salicon2017.zip)**

## Contact
For more datails about our research please visit our [page](http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=30).

If you have any general doubt about our work, please use the [public issues section](https://github.com/marcellacornia/sam/issues) on this github repo. Alternatively, drop us an e-mail at <marcella.cornia@unimore.it> or <lorenzo.baraldi@unimore.it>.
