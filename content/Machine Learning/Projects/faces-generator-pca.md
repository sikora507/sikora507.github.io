---
title: "Real time faces generator using TensorflowJs"
date: 2019-03-01
tags:
  - machine-learning
---

<a data-router-ignore href="/static/machine-learning/demos/faces-generator-pca/index.html">
Go to project page
</a>
<hr/>
<img class="img-thumbnail" src="Machine Learning/assets/projects/faces-generator-pca/faces-project-teaser.png"/>
<hr/>
<div class="justify mt-4">
<img class="pull-right ml-4 align-top" src="Machine Learning/assets/projects/faces-generator-pca/project-faces-diagram.png">
<p>
This project was made on linux machine with nvidia-docker. 
</p>
<p>
Jupyter notebook was used to write python code for tensorflow's keras api to create autoencoder. 
</p>
<p>
This autoencoder was then trained on over 13000 human faces to learn how to reproduce human faces from 90 inputs.
</p> 
<p>
Then principal component analysis was used to determine which features are most important. In the end, there was 27 principal components which can be converted to 90 input neurons of the decoder.
</p>
<p>
The decoder was extracted from autoencoder model and converted to TensorflowJS compatibile json format.
</p>
<p>
This allowed to create a simple website in VueJs with sliders which values represent every PCA component and display decoded images in html canvas.
</p>
</div>
