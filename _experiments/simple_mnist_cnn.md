---
layout: post
title:  "MNIST digits recognition using Keras and CNNs"
date:   2019-01-31 11:53:35 +0100
categories: image recognition convolution neural networks keras
---

This is a simple example of creating sequential convolutional neural network model in tensorflow keras and training it to recognize MNIST digits dataset that's today's "Hello World" of image recognition.

Import required libraries


```python
from tensorflow.keras import layers
from tensorflow.keras import models
```

Create Sequential model:


```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

Print out summary of the model:


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                36928     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 93,322
    Trainable params: 93,322
    Non-trainable params: 0
    _________________________________________________________________


Import MNIST dataset and to_categorical function


```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
```

Init train and test data


```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Reshape and normalize data and labels


```python
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

Train the model


```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
             metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
```

    Epoch 1/5
    60000/60000 [==============================] - 5s 77us/step - loss: 0.1711 - acc: 0.9471
    Epoch 2/5
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0488 - acc: 0.9849
    Epoch 3/5
    60000/60000 [==============================] - 4s 60us/step - loss: 0.0336 - acc: 0.9898
    Epoch 4/5
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0255 - acc: 0.9924
    Epoch 5/5
    60000/60000 [==============================] - 4s 59us/step - loss: 0.0196 - acc: 0.9940

    <tensorflow.python.keras.callbacks.History at 0x7f07860745f8>



Save trained model


```python
model.save('mnist_cnn.h5')
```

Load and evaluate the model (test the accuracy on test data)


```python
from tensorflow.keras.models import load_model
```


```python
model = load_model('mnist_cnn.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

    10000/10000 [==============================] - 0s 49us/step


Print out the accuracy


```python
test_acc
```




    0.9918



Show example test image:


```python
import matplotlib.pyplot as plt
```


```python
test_img = test_images[0].reshape(28,28)
plt.imshow(test_img, cmap='Greys')
```




    <matplotlib.image.AxesImage at 0x7f076011e5f8>




![png](/assets/img/outputs/output_24_1.png)


Predict example test image using the model


```python
prediction_probs = model.predict(test_img.reshape(1, 28,28,1)).flatten()
```


```python
prediction_probs.argmax()
```




    7



This experiment was made after reading Deep Learning with Python by Francois Chollet. I strongly recomment this book to everyone interested in python, deep learning, tensorflow and keras.

<a target="_blank"  href="https://www.amazon.com/gp/product/1617294438/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1617294438&linkCode=as2&tag=sikora507-20&linkId=99dea8b6292cc17dfeb589d5514bc37c"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1617294438&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=sikora507-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1617294438" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

<a target="_blank" href="https://www.amazon.com/gp/product/1617294438/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1617294438&linkCode=as2&tag=sikora507-20&linkId=7889193ba210dc35fd6c62da47e14df7">Deep Learning with Python</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1617294438" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

Source code is available to download at my github repository:

<https://github.com/sikora507/keras_mnist>