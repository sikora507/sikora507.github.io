---
layout: post
title:  "Heatmaps for CNNs with Grad-CAM process"
date:   2019-02-08 11:53:35 +0100
categories: GradCam CNN heatmap
---

Import required libraries


```python
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
```

List all images


```python
img_path = './images/'
images = sorted(os.listdir(img_path))
images = [img_path + img for img in images]
images
```




    ['./images/img1.jpg',
     './images/img10.jpg',
     './images/img2.jpg',
     './images/img3.jpg',
     './images/img4.jpg',
     './images/img5.jpg',
     './images/img6.jpg',
     './images/img7.jpg',
     './images/img8.jpg',
     './images/img9.jpg',
     './images/output_with_heatmap.jpg']



Import VGG16 model


```python
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet')
```

Preview sample image


```python
image_to_load = images[5]
```


```python
img = image.load_img(image_to_load, target_size=(224,224))
x = image.img_to_array(img)
plt.imshow(np.int32(x))
```




    <matplotlib.image.AxesImage at 0x7f9f96935d30>




![png](/assets/img/experiments/cnn_heatmap/output_8_1.png)


Preprocess input to be compatibile with VGG16 model (e.g. substract mean and divide by standard deviation)


```python
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
```

Predict what is on the picture using the model


```python
preds = model.predict(x)
prediction = decode_predictions(preds, top=1)[0][0]
print('Predicted: ', prediction[1], ' with probability ', prediction[2], '%', sep='')
print('Top 3 predictions:')
print(decode_predictions(preds, top=3)[0])
```

    Predicted: fox_squirrel with probability 0.9221031%
    Top 3 predictions:
    [('n02356798', 'fox_squirrel', 0.9221031), ('n02441942', 'weasel', 0.02617497), ('n02442845', 'mink', 0.025668971)]


Get index of the prediction in one-hot encoded output array


```python
max_ind = np.argmax(preds[0])
```


```python
preds[:,max_ind]
```




    array([0.9221031], dtype=float32)



Print out the model summary


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 224, 224, 3)       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 1000)              4097000   
    =================================================================
    Total params: 138,357,544
    Trainable params: 138,357,544
    Non-trainable params: 0
    _________________________________________________________________


Print out layer names


```python
[layer.name for layer in model.layers]
```




    ['input_1',
     'block1_conv1',
     'block1_conv2',
     'block1_pool',
     'block2_conv1',
     'block2_conv2',
     'block2_pool',
     'block3_conv1',
     'block3_conv2',
     'block3_conv3',
     'block3_pool',
     'block4_conv1',
     'block4_conv2',
     'block4_conv3',
     'block4_pool',
     'block5_conv1',
     'block5_conv2',
     'block5_conv3',
     'block5_pool',
     'flatten',
     'fc1',
     'fc2',
     'predictions']



Save last convolutional layer to variable


```python
last_conv_layer = model.get_layer('block5_conv3')
```

Import keras backend for operations needed to perform Grad-CAM process


```python
import tensorflow.keras.backend as K
```

Get output tensor for predicted class from model output


```python
model_output = model.output[:, max_ind]
```

Compute gradients of predicted class with regard to the output feature map of last convolution layer


```python
grads = K.gradients(model_output, last_conv_layer.output)[0]
print(grads)
```

    Tensor("gradients/block5_pool/MaxPool_grad/MaxPoolGrad:0", shape=(?, 14, 14, 512), dtype=float32)


It can be interpreted as 512 filters or features with size 14 by 14 pixels

Average all values from each filter together. Now each entry corresponds to mean intensity of the gradient over a specific feature map channel. 

In other words, every 14x14 filter will be now aweraged to single value.


```python
pooled_grads = K.mean(grads, axis=(0,1,2))
print(pooled_grads)
```

    Tensor("Mean:0", shape=(512,), dtype=float32)


<https://www.tensorflow.org/api_docs/python/tf/keras/backend/function>
"Function" lets us get custom tensors from the model. We don't want the model's final output but output from last convolution layer and pooled_grads.

We can do that by passing to this function our model input tensor and array of our outputs which will be evaluated after calling this function.


```python
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
```

Print out output from one of the 512 filters in last convolution layer:


```python
plt.imshow(conv_layer_output_value[:,:,10])
```




    <matplotlib.image.AxesImage at 0x7f9f947c8898>




![png](/assets/img/experiments/cnn_heatmap/output_34_1.png)


Multiply each channel in the feature map array with averaged gradient values corresponding to that channel. This resembles how important this channel is with regards to predicted class.


```python
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
```

Print out the same convolution layer channel output after scalling with pooled gradients:


```python
plt.imshow(conv_layer_output_value[:,:,10])
```




    <matplotlib.image.AxesImage at 0x7f9f94789d30>




![png](/assets/img/experiments/cnn_heatmap/output_38_1.png)


Create the heatmap


```python
conv_layer_output_value.shape
```




    (14, 14, 512)



Merge all 512 channels together by averaging all their pixel values:


```python
heatmap = np.mean(conv_layer_output_value, axis=-1)
```


```python
heatmap.shape
```




    (14, 14)



Scale and plot the heatmap


```python
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
```




    <matplotlib.image.AxesImage at 0x7f9f94750390>




![png](/assets/img/experiments/cnn_heatmap/output_45_1.png)


Import OpenCV to easily blend heatmap with original image


```python
import cv2
```

Save the image with heatmap applied


```python
img_cv = cv2.imread(image_to_load)
heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img_cv
cv2.imwrite('./images/output_with_heatmap.jpg', superimposed_img)
```




    True



Scale and plot the results in the notebook:


```python
max_val = superimposed_img.max()
# superimposed_img[...,::-1] is the trick to convert 
# opencv's BGR to matplotlib's RGB color channels
plt.imshow(np.int32(superimposed_img[...,::-1]/max_val*255))
```




    <matplotlib.image.AxesImage at 0x7f9f946ec198>




![png](/assets/img/experiments/cnn_heatmap/output_51_1.png)


This experiment was made after reading Deep Learning with Python by Francois Chollet. I strongly recomment this book to everyone interested in python, deep learning, tensorflow and keras.

<a target="_blank"  href="https://www.amazon.com/gp/product/1617294438/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1617294438&linkCode=as2&tag=sikora507-20&linkId=99dea8b6292cc17dfeb589d5514bc37c"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1617294438&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=sikora507-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1617294438" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

<a target="_blank" href="https://www.amazon.com/gp/product/1617294438/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1617294438&linkCode=as2&tag=sikora507-20&linkId=7889193ba210dc35fd6c62da47e14df7">Deep Learning with Python</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1617294438" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

Github URL:

<https://github.com/sikora507/image-classification-heatmap>