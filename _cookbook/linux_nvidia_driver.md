---
layout: post
title:  "Install nVidia graphics driver on ubuntu linux"
date:   2019-01-26 11:53:35 +0100
categories: nvidia driver linux
---
If you have strong nVidia graphics card and you want to run your machine learning experiments with tensorflow using your GPU and you are running linux, you'll need to have nVidia drivers installed.

There are many ways to install latest drivers for nVidia but after many trials here's my fastest way:

1. Add ppa repository and run apt update. This will allow us to find latest nVidia drivers
```
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
```

2. Open "Software & Updates" and go to "Additional Drivers" tab:
![Software & Updates](/assets/img/software_updates.png)
If you ran previous commands, you should be able to see a list of nVidia drivers.

3. Click on the latest driver (in my case nvidia-driver-415 (open source))

4. Wait untill driver has been installed. Done.
