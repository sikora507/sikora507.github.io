---
layout: post
title: "Faces autoencoder with PCA"
date: 2019-02-26 11:53:35 +0100
categories: GradCam CNN heatmap
---

This is an experiment that will be used to create real time face generator using autoencoders and principal component analysis.

Final solution will be a website with sliders that user can move around
to generate faces in real time in the browser using TensorflowJS.

This work is inspired by [carykh](https://github.com/carykh/alignedCelebFaces).
This version is rewritten in jupyter notebook using Tensorflow's Keras for model training as it is easier to export keras model to TensorflowJS format.

The purpose of this experiment is to create keras model of autoencoder that will be
converted to TensorflowJS version which can be run in the browser.

Github url: <https://github.com/sikora507/faces-autoencoder>

To run this experiment, you need to have docker and docker-compose installed.
Then you can run it by typing `docker-compose up` in the terminal.

```python
import os
import tensorflow as tf
import numpy as np
from scipy import misc
import random
import math
import face_recognition
from skimage import transform
import os.path
import urllib.request
import time
import requests
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
```

```python
%matplotlib inline
```

## Set global variables

```python
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
DENSE_SIZE = 200
```

### Set up variables and download helpers

```python
START_MONTH = 11
START_DATE = 17
START_SLOT = 6

DESIRED_X = 32
DESIRED_Y = 21
DESIRED_SIZE = 24

NAMES_PER_FILE = 100
TEMP_FILENAME = "temp.png"
monthOn = 0
dayOn = 0
daysInMonth = [31,29,31,30,31,30,31,31,30,31,30,31]
monthNames = ['january','february','march','april','may','june','july','august','september','october','november','december']

def getAvg(face, landmark):
    cum = np.zeros((2))
    for point in face[landmark]:
        cum[0] += point[0]
        cum[1] += point[1]
    return cum/len(face[landmark])

def getNorm(a):
    return (a-np.mean(a))/np.std(a)

```

## Download all the data

```python
imageCounter = 0
for monthOn in range(START_MONTH,12):
    thisStartDate = 1
    if monthOn == START_MONTH:
        thisStartDate = START_DATE
    for dayOn in range(thisStartDate,daysInMonth[monthOn]+1):
        response = urllib.request.urlopen("https://www.famousbirthdays.com/"+monthNames[monthOn]+str(dayOn)+".html")
        pageSource = response.read().splitlines()
        lineOn = 0
        while str(pageSource[lineOn]) != "b'<div class=\"container people-list\">'":
            lineOn += 1

        thisStartSlot = 0
        if monthOn == START_MONTH and dayOn == START_DATE:
            thisStartSlot = START_SLOT
        for slotOn in range(0,48):
            while "class=\"face person-item\"" not in str(pageSource[lineOn]):
                lineOn += 1
            iul = str(pageSource[lineOn])
            pnl = str(pageSource[lineOn+4])
            imageURL = iul[iul.index("background: url(")+16:iul.index(") no-repeat center center")]
            personName = ""
            age = ""
            pnl_s = 2
            if pnl[pnl_s] == ' ':
               pnl_s += 1
            if "," in pnl:
                personName = pnl[pnl_s:pnl.index(",")]
                age = pnl[pnl.index(",")+2:-1]
            else:
                personName = pnl[pnl_s:pnl.index("(")-1]
                age = pnl[pnl.index("(")+1:pnl.index(")")]

            if slotOn < thisStartSlot or imageURL == 'https://www.famousbirthdays.com/faces/large-default.jpg' or personName == "Ronan Domingo"  or personName == "Glam And Gore" or personName == "Edith Piaf" or personName == "Lexi Marie":
                print(personName+" SKIPPED!")
            else:

                img_data = requests.get(imageURL).content
                with open("temp.png", 'wb') as handler:
                    handler.write(img_data)

                image_face_info = face_recognition.load_image_file(TEMP_FILENAME)
                face_landmarks = face_recognition.face_landmarks(image_face_info)

                image_numpy = misc.imread(TEMP_FILENAME)
                colorAmount = 0
                if len(image_numpy.shape) == 3:
                    nR = getNorm(image_numpy[:,:,0])
                    nG = getNorm(image_numpy[:,:,1])
                    nB = getNorm(image_numpy[:,:,2])
                    colorAmount = np.mean(np.square(nR-nG))+np.mean(np.square(nR-nB))+np.mean(np.square(nG-nB))
                if len(face_landmarks) == 1 and colorAmount >= 0.04: # We need there to only be one face in the image, AND we need it to be a colored image.
                    leftEyePosition = getAvg(face_landmarks[0],'left_eye')
                    rightEyePosition = getAvg(face_landmarks[0],'right_eye')
                    nosePosition = getAvg(face_landmarks[0],'nose_tip')
                    mouthPosition = getAvg(face_landmarks[0],'bottom_lip')

                    centralPosition = (leftEyePosition+rightEyePosition)/2

                    faceWidth = np.linalg.norm(leftEyePosition-rightEyePosition)
                    faceHeight = np.linalg.norm(centralPosition-mouthPosition)
                    if faceWidth >= faceHeight*0.7 and faceWidth <= faceHeight*1.5:

                        faceSize = (faceWidth+faceHeight)/2

                        toScaleFactor = faceSize/DESIRED_SIZE
                        toXShift = (centralPosition[0])
                        toYShift = (centralPosition[1])
                        toRotateFactor = np.arctan2(rightEyePosition[1]-leftEyePosition[1],rightEyePosition[0]-leftEyePosition[0])

                        rotateT = transform.SimilarityTransform(scale=toScaleFactor,rotation=toRotateFactor,translation=(toXShift,toYShift))
                        moveT = transform.SimilarityTransform(scale=1,rotation=0,translation=(-DESIRED_X,-DESIRED_Y))

                        outputArr = transform.warp(image=image_numpy,inverse_map=(moveT+rotateT))[0:IMAGE_HEIGHT,0:IMAGE_WIDTH]

                        misc.imsave("../data/dataFace"+str(imageCounter)+".png",outputArr)
                        if imageCounter%NAMES_PER_FILE == 0:
                            f = open("../names/name"+str(imageCounter)+".txt","w+")
                        fame = str(slotOn)
                        if monthOn == 1 and dayOn == 29:
                            fame *= 4
                        f.write(personName+","+fame+","+age+"\n")
                        if imageCounter%NAMES_PER_FILE == (NAMES_PER_FILE-1):
                            f.close()
                        print("DAY "+monthNames[monthOn]+" "+str(dayOn)+":  I just used person "+personName+" to create image number "+str(imageCounter))
                        imageCounter += 1
                time.sleep(0.5)
            lineOn += 1

```

### Store images count

```python
images_count = len(os.listdir('../data/'))
```

### Setup helper functions

```python
def CreateAutoEncoder():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))  # adapt this if using `channels_first` image data format

    x = Conv2D(40, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(80, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(150, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(200, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(DENSE_SIZE, activation='relu')(x)

    y = Dense(3200, activation='relu')(encoded)
    y = Reshape([4,4,200])(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(150, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(80, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(40, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(15, (3, 3), activation='relu', padding='same')(y)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(y)

    autoencoder = Model(input_img, decoded)
    #optimizer = tf.keras.optimizers.Adam(lr=0.005)
    autoencoder.compile(optimizer='adagrad', loss='binary_crossentropy')
    return autoencoder
```

```python
def WriteOutputFileResult(epoch):
    if epoch%10!=0:
        return

    imageIndex = np.random.randint(images_count)
    output = autoencoder.predict(batch_data[imageIndex].reshape(1,64,64,3))
    exampleImage = np.empty([IMAGE_HEIGHT*2,IMAGE_WIDTH,3])
    exampleImage[0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:3] = batch_data[imageIndex,0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:3]
    exampleImage[IMAGE_HEIGHT:IMAGE_HEIGHT*2,0:IMAGE_WIDTH,0:3] = output[0,0:IMAGE_HEIGHT,0:IMAGE_WIDTH,0:3]
    exampleImage = np.clip(exampleImage, 0, 1)
    misc.imsave('../modelExamples/encoder'+str(epoch)+'.png',exampleImage)

    save_path='../models/keras.h5'
    autoencoder.save(save_path)
    print("MODEL SAVED: "+str(save_path))
```

## Train autoencoder

```python
from tensorflow.keras.models import load_model
modelPath = '../models/keras.h5'
if os.path.isfile(modelPath):
    print('Loading existing model')
    autoencoder = load_model('../models/keras.h5', custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})
else:
    print('Creating new model')
    autoencoder = CreateAutoEncoder()
batch_data = np.empty([images_count,IMAGE_HEIGHT,IMAGE_WIDTH,3])
for example in range(images_count):
    imageIndex = example# +  int(math.floor(random.randrange(images_count)))
    imagio = misc.imread('../data/dataFace'+str(imageIndex)+'.png')
    batch_data[example] = imagio[:,:,0:3]/255.0
autoencoder.fit(batch_data, batch_data,
    epochs=10000,
    batch_size=100,
    shuffle=True,
    #validation_data=(batch_data, batch_data),
    callbacks=[
        #tf.keras.callbacks.ModelCheckpoint('../models/keras.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: WriteOutputFileResult(epoch))
    ])


```

## Tear apart encoder and decoder

```python
[layer.name for layer in autoencoder.layers]
```

    ['input_3',
     'conv2d_18',
     'max_pooling2d_8',
     'conv2d_19',
     'max_pooling2d_9',
     'conv2d_20',
     'max_pooling2d_10',
     'conv2d_21',
     'max_pooling2d_11',
     'flatten_2',
     'dense_4',
     'dense_5',
     'reshape_2',
     'up_sampling2d_8',
     'conv2d_22',
     'up_sampling2d_9',
     'conv2d_23',
     'up_sampling2d_10',
     'conv2d_24',
     'up_sampling2d_11',
     'conv2d_25',
     'conv2d_26']

```python
encoder = Sequential()
for i in range(0,11):
    encoder.add(autoencoder.layers[i])
encoder.build()

decoder = Sequential()
for i in range(11,len(autoencoder.layers)):
    decoder.add(autoencoder.layers[i])

decoder.build(input_shape=(None, 200))
```

```python
def getEncodedImage(index):
    image = misc.imread('../data/dataFace'+str(index)+'.png')
    image = image[:,:,0:3]/255.0
    return image
```

### Preview random image to be encoded

```python
randIndex = np.random.randint(0, images_count)
image = getEncodedImage(randIndex)
plt.imshow(image)
```

    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:2: DeprecationWarning: `imread` is deprecated!
    `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``imageio.imread`` instead.






    <matplotlib.image.AxesImage at 0x7faf480ad128>

![png](/assets/img/experiments/faces_autoencoder/output_20_2.png)

```python
encoded = encoder.predict(image.reshape([-1, image.shape[0], image.shape[1], image.shape[2]]))
print('Encoded image:\n' ,encoded[0])
```

    Encoded image:
     [0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         1.6740656  0.
     0.         1.2446964  0.         0.         4.6464124  2.7161005
     4.744044   0.         0.         0.         1.0253751  0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.516001   0.         0.         0.         0.         0.
     0.         0.         0.         0.         2.0416033  0.
     1.2738434  0.         0.         0.         4.646125   0.
     2.3047757  0.         0.         0.6224123  4.424849   0.
     0.         1.1240004  0.         0.         0.         1.548488
     0.         0.4614426  0.         0.         0.         0.
     0.         1.1851753  1.2601664  0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.5733514  0.         0.         0.         0.         0.18939778
     1.6195832  4.0283566  0.         0.         0.         1.4311539
     0.         1.9477886  1.0686957  0.         0.         1.4714222
     0.         0.         0.         0.         0.         0.
     0.         1.5717533  0.         0.         0.         1.5706655
     0.         0.         4.989835   0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.         0.
     0.87802136 0.         0.         0.         0.         0.
     0.         0.         0.         1.3519242  0.         0.
     2.0465598  3.8467922  0.         0.         0.17653233 0.
     2.6257434  1.0397722  0.         0.         1.2316633  0.
     0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         1.5195677  0.
     0.         0.         0.         1.062976   0.         0.99829024
     2.1957307  0.         3.938364   0.         0.5726944  0.
     0.         0.         0.         0.5094159  0.         0.
     0.         0.         0.         0.         4.072344   0.
     0.         0.         0.         0.         3.8419755  0.
     0.         0.        ]

```python
decoded = decoder.predict(encoded)
```

### Recreate image from encoded state

```python
plt.imshow(decoded[0])
```

    <matplotlib.image.AxesImage at 0x7faefc7a5cc0>

![png](/assets/img/experiments/faces_autoencoder/output_24_1.png)

## Create dataframe to inspect all encoded images together

```python
import pandas as pd
```

```python
all_rows = []
for i in range(images_count):
    image = getEncodedImage(i)
    encoded_internal = encoder.predict(image.reshape([-1, image.shape[0], image.shape[1], image.shape[2]]))[0]
    all_rows.append(encoded_internal)
df = pd.DataFrame(all_rows)
df.describe()

```

    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:2: DeprecationWarning: `imread` is deprecated!
    `imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
    Use ``imageio.imread`` instead.

<div style="overflow-x:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>190</th>
      <th>191</th>
      <th>192</th>
      <th>193</th>
      <th>194</th>
      <th>195</th>
      <th>196</th>
      <th>197</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>527.0</td>
      <td>527.000000</td>
      <td>527.0</td>
      <td>527.000000</td>
      <td>527.0</td>
      <td>527.0</td>
      <td>527.0</td>
      <td>527.000000</td>
      <td>527.0</td>
      <td>527.0</td>
      <td>...</td>
      <td>527.000000</td>
      <td>527.0</td>
      <td>527.0</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.0</td>
      <td>527.000000</td>
      <td>527.0</td>
      <td>527.000000</td>
      <td>527.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0</td>
      <td>0.383350</td>
      <td>0.0</td>
      <td>0.497579</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.563971</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.588612</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.292458</td>
      <td>0.928342</td>
      <td>0.0</td>
      <td>0.658084</td>
      <td>0.0</td>
      <td>0.906207</td>
      <td>0.520338</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>1.190903</td>
      <td>0.0</td>
      <td>1.101257</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.455407</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.390174</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.263822</td>
      <td>1.070300</td>
      <td>0.0</td>
      <td>1.130944</td>
      <td>0.0</td>
      <td>1.259891</td>
      <td>1.474352</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.294710</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.333912</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.655429</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.324040</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.328242</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.321734</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.510635</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.580515</td>
      <td>0.0</td>
      <td>0.958642</td>
      <td>0.0</td>
      <td>1.432640</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0</td>
      <td>7.928043</td>
      <td>0.0</td>
      <td>8.070414</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.701365</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>12.323253</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14.501927</td>
      <td>5.239333</td>
      <td>0.0</td>
      <td>6.052303</td>
      <td>0.0</td>
      <td>6.644118</td>
      <td>9.616698</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 200 columns</p>
</div>

## Save mean from encodings to file

```python

mean = df.mean().values.tolist()

text_file = open("../models/mean.json", "w")
text_file.write(json.dumps(mean))
text_file.close()
```

## Filter out columns all containing zeros

```python
m2 = df.loc[:, (df != 0).any(axis=0)]
```

```python
m2.describe()
```

<div style="overflow-x:scroll">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>3</th>
      <th>7</th>
      <th>10</th>
      <th>13</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>21</th>
      <th>...</th>
      <th>179</th>
      <th>180</th>
      <th>183</th>
      <th>184</th>
      <th>190</th>
      <th>193</th>
      <th>194</th>
      <th>196</th>
      <th>198</th>
      <th>199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>...</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
      <td>527.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383350</td>
      <td>0.497579</td>
      <td>1.563971</td>
      <td>1.032203</td>
      <td>0.645219</td>
      <td>1.415085</td>
      <td>0.818090</td>
      <td>0.795896</td>
      <td>1.025534</td>
      <td>0.855816</td>
      <td>...</td>
      <td>0.830482</td>
      <td>0.720698</td>
      <td>0.921920</td>
      <td>1.092613</td>
      <td>0.588612</td>
      <td>0.292458</td>
      <td>0.928342</td>
      <td>0.658084</td>
      <td>0.906207</td>
      <td>0.520338</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.190903</td>
      <td>1.101257</td>
      <td>1.455407</td>
      <td>1.125508</td>
      <td>1.383625</td>
      <td>1.554425</td>
      <td>1.545218</td>
      <td>1.385235</td>
      <td>1.114971</td>
      <td>1.103555</td>
      <td>...</td>
      <td>1.060839</td>
      <td>1.136248</td>
      <td>1.201186</td>
      <td>1.640923</td>
      <td>1.390174</td>
      <td>1.263822</td>
      <td>1.070300</td>
      <td>1.130944</td>
      <td>1.259891</td>
      <td>1.474352</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.294710</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.333912</td>
      <td>0.756519</td>
      <td>0.000000</td>
      <td>1.057168</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.758654</td>
      <td>0.457324</td>
      <td>...</td>
      <td>0.382845</td>
      <td>0.000000</td>
      <td>0.473265</td>
      <td>0.104616</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.655429</td>
      <td>0.000000</td>
      <td>0.324040</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.328242</td>
      <td>2.321734</td>
      <td>1.556919</td>
      <td>0.611775</td>
      <td>2.378852</td>
      <td>1.168464</td>
      <td>1.147271</td>
      <td>1.744949</td>
      <td>1.374141</td>
      <td>...</td>
      <td>1.381343</td>
      <td>1.199598</td>
      <td>1.443289</td>
      <td>1.651781</td>
      <td>0.510635</td>
      <td>0.000000</td>
      <td>1.580515</td>
      <td>0.958642</td>
      <td>1.432640</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.928043</td>
      <td>8.070414</td>
      <td>8.701365</td>
      <td>5.879054</td>
      <td>10.312868</td>
      <td>6.995349</td>
      <td>9.534487</td>
      <td>8.131895</td>
      <td>6.139172</td>
      <td>6.481678</td>
      <td>...</td>
      <td>5.561953</td>
      <td>6.141798</td>
      <td>6.244396</td>
      <td>9.122747</td>
      <td>12.323253</td>
      <td>14.501927</td>
      <td>5.239333</td>
      <td>6.052303</td>
      <td>6.644118</td>
      <td>9.616698</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 107 columns</p>
</div>

```python
pca_columns = len(m2.columns)
```

### This number shows how many principal components are needed

#### We can take only non-zero columns from our encodings table

This will allow to use fewer components

```python
pca_columns
```

    107

## Manual PCA

```python
np_all_rows = np.asarray(all_rows)

pca = PCA(n_components=pca_columns)
pca.fit(np_all_rows)
values = np.sqrt(pca.explained_variance_)
vectors = pca.components_

json_vectors = json.dumps(vectors.tolist())
json_values = json.dumps(values.tolist())

all_rows_pca = pca.transform(np_all_rows)

sliders_min = all_rows_pca.min(axis=0)
sliders_max = all_rows_pca.max(axis=0)
json_sliders_min = json.dumps(sliders_min.tolist())
json_sliders_max = json.dumps(sliders_max.tolist())

text_file = open("../models/eigenVectors.json", "w")
text_file.write(json_vectors)
text_file.close()

text_file = open("../models/eigenValues.json", "w")
text_file.write(json_values)
text_file.close()

text_file = open("../models/slidersMin.json", "w")
text_file.write(json_sliders_min)
text_file.close()

text_file = open("../models/slidersMax.json", "w")
text_file.write(json_sliders_max)
text_file.close()
```

## Load data as it will be loaded by tensorflowJS in the browser

```python
with open('../models/eigenVectors.json') as f:
    eigenVectors = np.asarray(json.load(f))
with open('../models/eigenValues.json') as f:
    eigenValues = np.asarray(json.load(f))
with open('../models/mean.json') as f:
    meanData = np.asarray(json.load(f))
```

```python
eigenvectorInverses = np.linalg.pinv(eigenVectors)
```

```python
traits = encoded[0] - meanData
image_pca = np.matmul(traits, eigenvectorInverses) / eigenValues
```

```python
image_pca
```

    array([-1.21494064e+00, -4.95205738e-01,  1.58773161e+00,  1.03317212e+00,
            1.64704590e-02,  8.42720705e-02,  6.85427567e-02, -2.79387655e+00,
            1.87637418e+00,  7.37616443e-02,  2.54778020e-01, -1.77222121e-01,
            1.13375160e+00, -1.30607305e-01, -5.21181651e-01,  1.04267917e+00,
           -5.30473454e-01, -9.19137177e-01, -6.33939878e-01,  1.90321028e+00,
            1.06104141e+00, -7.84540597e-01,  6.10745502e-01,  7.75544652e-01,
           -3.78812174e-01, -8.48271968e-01, -9.15274639e-01, -5.84614425e-02,
            4.95471393e-01, -8.18511609e-01,  1.17733230e-01,  7.96533237e-01,
            3.81170546e-01, -7.34867326e-01, -1.80153036e-01,  2.69993923e-01,
           -4.40699800e-01,  4.58954842e-01,  1.20221894e+00,  1.05707949e+00,
           -9.26709735e-01, -2.62151517e-01,  4.12873764e-01,  9.51895180e-01,
           -1.82035008e-01,  2.05909696e-01, -3.62205891e-02,  1.64909150e-01,
            3.50870550e-02, -6.51322007e-01, -1.10071462e+00,  1.10239015e-01,
            1.00940874e+00, -3.22872511e-01, -6.86396836e-01, -3.72447165e-01,
            1.11109849e+00,  6.57288975e-01, -1.00576561e-01,  1.89648021e+00,
            3.43962315e-01,  2.34161901e-01,  3.68600332e-01,  8.69722112e-01,
            8.62145163e-01,  9.93404808e-01, -1.97469710e-01,  7.98476353e-01,
           -1.10964469e+00, -4.45596697e-01, -5.13701822e-01,  7.46462772e-01,
           -1.43335627e+00, -5.69662544e-01, -5.31171657e-01, -6.39782067e-01,
           -2.70193727e-01, -9.27538014e-01,  6.65862011e-02, -3.60068660e-01,
            1.08171159e+00, -2.99478694e-01,  8.84023699e-01,  4.59543898e-01,
            9.50298384e-01, -7.86344394e-02, -1.28323107e+00,  1.15862300e+00,
            7.38062574e-01, -8.21171997e-01,  1.04588902e+00, -1.54296072e+00,
            9.66833325e-01, -4.98282965e-01,  8.00057643e-01, -3.95012717e-01,
            9.15517991e-01, -1.49782181e+00,  5.22660921e-01,  5.27705578e-01,
            1.26221635e-03,  1.14477946e+00,  1.60914136e+00,  9.85090220e-01,
           -7.20066725e-01, -7.78477121e-01, -6.47627714e-01])

```python
recovered_from_pca = meanData.copy()
for i in range(pca_columns):
    recovered_from_pca += image_pca[i] * eigenValues[i] * eigenVectors[i]
```

```python
recovered_from_pca
```

    array([-8.30572673e-07, -8.26042793e-15,  7.28689154e-07, -1.38158148e-13,
           -4.69476584e-07, -1.24504785e-07, -4.88584873e-07,  1.88553600e-13,
            1.75781359e-07,  8.27562621e-07,  1.67406559e+00,  2.58648698e-07,
           -1.61953981e-07,  1.24469638e+00, -1.84599036e-07,  3.55158112e-07,
            4.64641237e+00,  2.71610045e+00,  4.74404383e+00,  1.98366497e-13,
           -1.74709958e-07, -7.35730921e-14,  1.02537513e+00,  5.83238056e-07,
           -1.10047388e-13,  3.03989825e-07, -1.23132841e-13, -3.87376862e-07,
            1.58832694e-07, -2.72547013e-07,  3.58801530e-14,  1.76143822e-14,
           -4.30501988e-14, -2.59269578e-07, -2.51543659e-07,  9.44383460e-14,
            5.16000986e-01,  2.46116744e-07,  3.67596578e-14,  6.68791436e-07,
           -3.52233793e-08,  4.35242120e-14, -3.67191394e-08,  1.43867990e-13,
            2.98658325e-07,  3.94759415e-08,  2.04160333e+00, -1.50137861e-07,
            1.27384341e+00, -1.00947029e-13, -1.36277723e-07, -1.51687690e-13,
            4.64612484e+00, -1.96762745e-13,  2.30477571e+00, -3.33317077e-08,
           -2.60209557e-07,  6.22412324e-01,  4.42484903e+00, -3.28825352e-08,
           -3.63440664e-08,  1.12400043e+00, -6.86406919e-08, -2.03170814e-14,
            1.06426262e-07,  1.54848802e+00,  1.47350922e-07,  4.61442590e-01,
            1.71426241e-13, -8.23370235e-14, -1.34194914e-07,  6.22532487e-08,
           -1.05264673e-07,  1.18517530e+00,  1.26016641e+00,  3.41220108e-14,
            5.01983140e-08, -1.56444773e-07,  8.23085090e-14,  1.12135995e-13,
           -2.42597609e-13,  1.87335794e-08,  2.74907211e-08, -1.98532163e-13,
            5.73351383e-01, -2.23147896e-08,  1.45389737e-07,  1.52400662e-13,
            2.31082514e-13,  1.89397782e-01,  1.61958325e+00,  4.02835655e+00,
           -3.84828551e-08, -2.53931234e-08, -5.17085792e-08,  1.43115389e+00,
           -7.45952721e-08,  1.94778860e+00,  1.06869566e+00, -9.69328784e-14,
            3.07462389e-14,  1.47142220e+00,  1.67131066e-13,  1.36094738e-07,
            1.29731295e-13, -6.22669381e-09, -3.90126941e-08,  3.06171007e-09,
            1.82071362e-08,  1.57175326e+00,  2.47293505e-13, -2.96403527e-14,
            0.00000000e+00,  1.57066548e+00,  0.00000000e+00,  0.00000000e+00,
            4.98983479e+00,  0.00000000e+00,  4.21225554e-14,  9.77273817e-14,
            1.91644877e-13,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  1.33545952e-13,  0.00000000e+00,
            0.00000000e+00, -1.43735718e-13, -2.28697269e-14,  0.00000000e+00,
            8.78021359e-01,  0.00000000e+00,  0.00000000e+00, -7.04210995e-15,
            0.00000000e+00,  0.00000000e+00, -5.41233725e-15, -1.47889947e-13,
            0.00000000e+00,  1.35192418e+00,  0.00000000e+00, -3.06933298e-14,
            2.04655981e+00,  3.84679222e+00,  0.00000000e+00,  0.00000000e+00,
            1.76532328e-01,  0.00000000e+00,  2.62574339e+00,  1.03977215e+00,
            2.33606537e-13, -4.60990024e-14,  1.23166335e+00,  0.00000000e+00,
            0.00000000e+00, -1.22700461e-13,  0.00000000e+00,  7.23830718e-14,
            0.00000000e+00,  1.28057287e-14,  6.03857242e-14,  0.00000000e+00,
           -3.15511506e-14,  0.00000000e+00,  1.51956773e+00,  2.93515212e-14,
            0.00000000e+00,  1.56125113e-14, -2.00616569e-14,  1.06297600e+00,
            0.00000000e+00,  9.98290241e-01,  2.19573069e+00,  5.76066972e-14,
            3.93836403e+00,  0.00000000e+00,  5.72694421e-01,  3.06946742e-13,
            2.72847717e-13,  0.00000000e+00,  0.00000000e+00,  5.09415925e-01,
           -7.86610360e-15,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  4.07234383e+00,  0.00000000e+00,
            0.00000000e+00, -2.00222976e-13, -5.84567117e-14,  0.00000000e+00,
            3.84197545e+00,  0.00000000e+00,  3.00013486e-13, -6.78797296e-15])

```python
decoded = decoder.predict(recovered_from_pca.reshape(1,DENSE_SIZE))
plt.imshow(decoded[0])
```

    <matplotlib.image.AxesImage at 0x7faefc6df1d0>

![png](/assets/img/experiments/faces_autoencoder/output_45_1.png)

### Save all images as encodings in json file

```python
json_all_rows_pca = json.dumps(all_rows_pca.tolist())

text_file = open("../models/facesPca.json", "w")
text_file.write(json_all_rows_pca)
text_file.close()
```

This experiment was made after reading Hands-On Machine Learning with Scikit-Learn and TensorFlow. It has principal component analysis well covered.

<a target="_blank"  href="https://www.amazon.com/gp/product/1491962291/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491962291&linkCode=as2&tag=sikora507-20&linkId=2afda09226851f789f24d5671b6d1ca3"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1491962291&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=sikora507-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1491962291" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

<a target="_blank" href="https://www.amazon.com/gp/product/1491962291/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491962291&linkCode=as2&tag=sikora507-20&linkId=64bd5103ef0b10b13cba3936d277e786">Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems</a><img src="//ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1491962291" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
