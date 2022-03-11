---
layout: post
title:  "0-Introduction"
date:   2022-03-11 14:09:09 +0800
category: Vision2022Posts
---



# Introduction

&copy; 王小龙 3180105098@zju.edu.cn

&copy; 李浩东 3190104890@zju.edu.cn

## Basic Information

- ***Smart Factory Soft Machine Vision***
- Smart Factory Innovation Club of Zhejiang University
- Class locations: 月牙楼301
- Class time: Sunday 9:30 to 11:30
- Number of students enrolled: 40

## Course Outline

- Image basics: pixels, colors, image formats
- Image processing techniques: filtering, binarization, cutting, morphological transformation, scale and rotation transformation, image gradient
- Image pattern recognition: line and circle detection, feature point detection, edge detection, `Blob` detection, feature point detection, pattern recognition

- Neural Network Basics: Neuron Structure, Multilayer Perceptron, Loss Function, Gradient Descent, `Back Propagation`
- Neural network basics: `Softmax` regression, deep neural networks, convolutional neural networks
- Neural network foundation: recurrent neural network, `NLP` natural language processing, cloud server resources
- Big Homework and Answers: Chessboard Recognition System

## Configuration

- Python [link](https://www.python.org/)
- If you have installed Python 3.7 and above, you can choose not to reinstall
- If Python has not been installed or the version is lower than 3.7, please reinstall it


![py_0]({{ site.url }}/images/Vision2022/py_install_1.png)

![py_1]({{ site.url }}/images/Vision2022/py_install_2.png)

![py_2]({{ site.url }}/images/Vision2022/py_install_3.png)

![py_3]({{ site.url }}/images/Vision2022/py_install_4.png)

```python
PS F:\> F:\Python310\python.exe
Python 3.10.2 (tags/v3.10.2:a58ebcc, Jan 17 2022, 14:12:15) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
```

- jupyter notebook or jupyter lab [link](https://jupyter.org/install)
    - `python.exe -m pip install jupyterlab`
    - `jupyter-lab`
    - `python.exe -m pip install notebook`
    - `jupyter notebook`
    - `python.exe -m pip install RISE`

- OpenCV [link](https://pypi.org/project/opencv-python/)
    - `python.exe -m pip install opencv-python`
- Required packages
    - `python.exe -m pip install -r requirements.txt`
- VS Code [link](https://code.visualstudio.com/)
    - Config python3.10.2 in your vscode

## Test configuration

- `python.exe .\test_opencv.py`
- `../images/cat.jpg`
- Press `Esc` or `q` on the keyboard to close the window


```python
import sys

print(sys.version)
```

    3.9.9 (tags/v3.9.9:ccb0e6a, Nov 15 2021, 18:08:50) [MSC v.1929 64 bit (AMD64)]



```python
import cv2
import numpy as np
import matplotlib.colors as mat_color

print(cv2.__version__)
path = "./images/cat.jpg"
img_bgr = cv2.imread(path)
no_norm = mat_color.Normalize(vmin=0, vmax=255, clip=False)
print(type(img_bgr))
print(np.shape(img_bgr))
```

    4.5.5
    <class 'numpy.ndarray'>
    (493, 493, 3)



```python
from matplotlib import pyplot as plt

plt.imshow(img_bgr, norm=no_norm)
```




    <matplotlib.image.AxesImage at 0x1a07e4b1580>




​    
![png]({{ site.url }}/images/Vision2022/0/output_15_1.png)
​    



```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb, norm=no_norm)
```




    <matplotlib.image.AxesImage at 0x1a07e5a2d60>




​    
![png]({{ site.url }}/images/Vision2022/0/output_16_1.png)
​    



```python
import os

os.makedirs('data', exist_ok=True)
data_file = os.path.join('data', 'cat.csv')
print(data_file)
with open(data_file, 'w') as f:
    f.write('R,G,B\n')
    for row in img_rgb:
        for rgb in row:
            f.write(str(rgb[0]) + ',' + str(rgb[1]) + ',' + str(rgb[2]) + '\n')
```

    data\cat.csv


## The End

2022.3
