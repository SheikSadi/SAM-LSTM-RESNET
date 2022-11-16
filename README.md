
# Smart Cropping Images based on Saliency Mapping 
This repository contains the reference code written in *Python 3* for **generating saliency maps** of images using Convolutional LSTM Resnet (implemented with *TensorFlow 2*)  and **smartly cropping** images based on these maps.
## Demo
<table>
	<tr>
	 <th>Original Image</th>
	 <th>Saliency Map</th>
	 <th>Smart Cropping</th>
	</tr>
  <tr>
	 <th><img src=https://user-images.githubusercontent.com/34588815/202250249-5282138e-2bfd-420a-9b84-15f7e68b9329.jpg></th>
	 <th><img src=https://user-images.githubusercontent.com/34588815/202250488-9121d697-98a5-47b1-b67a-a87c7c85b6ce.jpg></th>
	 <th><img src=https://user-images.githubusercontent.com/34588815/202250750-e594ef64-022d-4092-babf-fcbf60df2809.jpg></th>
	</tr>
</table>

## Getting Started
### [TRY IT NOW on Google Colab](./Smart_Cropping_Images_based_on_Saliency_Mapping.ipynb)
### Pip Installation
`pip install sam-lstm==1.0.0`
#### Dependencies
- Tensorflow 2.9.0
- Scipy 1.9.3
- Scikit Image 0.19.3
- OpenCV 2.9.0
- CUDA (GPU)

***Tips**: Building up the environment on your local machine from scratch can take hours. If you want to get your hands on asap, then just use Google Colab with GPU runtime. It's free and all these libraries are preinstalled there.*
***Note** It's mandatory to run the code on GPU runtime, otherwise it will fail. In a future release, the code will be made compatible with CPU runtime as well.*

### All you need is two lines!
```python
# Create a folder "samples" in the current directory
# Upload some images (.jpg, .png) in it
from sam_lstm import SalMap
SalMap.auto()
```
With just this two lines, `sam_lstm` will compile the  LSTM-based Saliency Attentive Convolutional model, generate raw saliency mapping in the **maps** folder, colored overlay mapping in the **cmaps** folder, bounding boxes over the images in the **boxes** and cropped ones in the **crops** folder. All of these will happen automatically. Just make sure you have .jpg/.jpeg/.png images in the **samples** folder.
### Training the weights
```python
from sal_lstm import SalMap

checkpoint = "/content/drive/MyDrive/Checkpoints/"

# Uncomment these lines if on GOOGLE COLAB
# import os
# from google.colab import drive
# drive.mount('/content/drive')
# if  not os.path.exists(checkpoint):
#	os.mkdir(checkpoint)

s = SalMap()
s.compile()
s.load_weights()
s.train("dataset", checkpoint, steps_per_epoch=1000)
```
With these line, you can start training the models using the Salicon 2017 dataset (which will get downloaded in the `dataset` directory)

## Credits
This work has been built on top of the following works:
1. [Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model by Cornia et. el. 2018](http://aimagelab.ing.unimore.it/imagelab/pubblicazioni/2018-tip.pdf)
2. Python 2 implementation (using Keras+Theano) by @marcellacornia. Check [here](https://github.com/marcellacornia/sam)

## Scope of work done by @SheikSadi
1. Implement the source code on Python 3, using latest versions (by November 2022) of tensorflow and opencv. The original work by @marcellacornia was written with Python2 and used Theano backend for Keras, all of which are now unsupported by the community.
2. Update the preprocessing stage to be compatible with Salicon 2017 dataset.
3. Convert the work into an open source Python package readily installable from PyPa.
4.  Addition of the `cropping` module that allows for smart cropping of images. I have written a [Descent from Hilltop](https://gist.github.com/SheikSadi/e107c42f88a67c4113e7ca587dc3e3ce) algorithm for finding the bounding boxes by which the images are cropped.

## The Underlying Neural Network 

![image](https://user-images.githubusercontent.com/34588815/196414378-34a16d32-9ac0-4f98-a287-18e4456e8d26.png)
## Resources
1. Training and validation dataset
- images: https://github.com/SheikSadi/SAM-LSTM-RESNET/releases/download/1.0.0/images.zip
- maps: https://github.com/SheikSadi/SAM-LSTM-RESNET/releases/download/1.0.0/maps.zip
- fixations: https://github.com/SheikSadi/SAM-LSTM-RESNET/releases/download/1.0.0/fixations.zip
2. No Top Resnet50 weights (NCHW format)
- https://github.com/SheikSadi/SAM-LSTM-RESNET/releases/download/1.0.0/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
3. Pre-trained weights
- trained by @marcellacornia: https://github.com/SheikSadi/SAM-LSTM-RESNET/releases/download/1.0.0/sam-resnet_salicon_weights.pkl
- trained by @SheikSadi on Google Colab:  https://github.com/SheikSadi/SAM-LSTM-RESNET/releases/download/1.0.0/sam-resnet-salicon.h5
