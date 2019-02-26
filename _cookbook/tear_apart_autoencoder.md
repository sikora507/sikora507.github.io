---
layout: post
title: "Separate encoder and decoder from trained autoencoder model"
date: 2019-02-26 11:53:35 +0100
categories: autoencoder separate encoder decoder
---

Let's say there's a autoencoder model with this structure:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_3 (InputLayer)         (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_18 (Conv2D)           (None, 64, 64, 40)        1120
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 32, 32, 40)        0
_________________________________________________________________
conv2d_19 (Conv2D)           (None, 32, 32, 80)        28880
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 16, 16, 80)        0
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 16, 16, 150)       108150
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 8, 8, 150)         0
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 8, 8, 200)         270200
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 4, 4, 200)         0
_________________________________________________________________
flatten_2 (Flatten)          (None, 3200)              0
_________________________________________________________________
dense_4 (Dense)              (None, 200)               640200
_________________________________________________________________
dense_5 (Dense)              (None, 3200)              643200
_________________________________________________________________
reshape_2 (Reshape)          (None, 4, 4, 200)         0
_________________________________________________________________
up_sampling2d_8 (UpSampling2 (None, 8, 8, 200)         0
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 8, 8, 150)         270150
_________________________________________________________________
up_sampling2d_9 (UpSampling2 (None, 16, 16, 150)       0
_________________________________________________________________
conv2d_23 (Conv2D)           (None, 16, 16, 80)        108080
_________________________________________________________________
up_sampling2d_10 (UpSampling (None, 32, 32, 80)        0
_________________________________________________________________
conv2d_24 (Conv2D)           (None, 32, 32, 40)        28840
_________________________________________________________________
up_sampling2d_11 (UpSampling (None, 64, 64, 40)        0
_________________________________________________________________
conv2d_25 (Conv2D)           (None, 64, 64, 15)        5415
_________________________________________________________________
conv2d_26 (Conv2D)           (None, 64, 64, 3)         408
=================================================================
Total params: 2,104,643
Trainable params: 2,104,643
Non-trainable params: 0
_________________________________________________________________
```

Dense layer in the middle (dense_4 (Dense) with 200 placeholders) acts as an output from encoder and at the same time as the input of decoder.

It's easy to separate encoder:

```python
encoder = Sequential()
for i in range(0,11):
    encoder.add(autoencoder.layers[i])
encoder.build()
```

But to build decoder, additional information about input dimensions needs to be set:

```python
decoder = Sequential()
for i in range(11,len(autoencoder.layers)):
    decoder.add(autoencoder.layers[i])

decoder.build(input_shape=(None, 200))
```
