 计算机视觉小班（Computer Vision） 刘家瑛

Email: liujiaying@pku.edu.cn
[Group](https://www.wict.pku.edu.cn/struct/ "小组")
助教：

- 汪文婧[Leading] doooshee@pku.edu.cn
- 王德昭，16本20硕 wangdz@pku.edu.cn

# 课程引论

## Research Interests

- **Image Enhancement 基于视网膜模型的低光照增强技术**
- **~~RetiNet技术~~用于图像处理**
- **Image Restoration 恶劣天气场景建模与复原**
- **Image Stylization 图像/视频风格化**
- **Video Action Analytics 多模态行为视频数据库PKUMMD**

## What is Computer Vision?

### Related Areas

Vision is the process of discovering from images *what is present in the world*, and where it is.
Nobel Prize  
Model Learning: An example of two-class  
High Diversity if Images  
Computer Vision: 50 Years of Progress  
总览

- [x] 判别问题（discriminative model）：让计算机认识世界（图像识别、物体检测、语义分别……）
- [x] 生成问题（）：让计算机理解世界（图像风格化……）
- [x] 他山之石，可以攻玉（领域交叉）
- [x] 顶会（CVPR/ICCV/ECCV）

## 评分

小班部分 50%

- 小班作业：20%，**10分**
- 大班作业：20%，**10分**
- 期末Project：60%，**30分**

大班部分 50%

- 期中考试
- 期末考试

Overall Philosophy

Artificial Intelligence Index report

# §1 Python&OpenCV 汪文婧

## 图像文件保存

```python
import cv2
import numpy as np

img = cv2.imread('example.jpg')

print(img.shape)
print(img.dtype)
norm_img = img.astype(np.float64) / 255.
light_norm_img = np.power(norm_img, 0.5)
light_img = light_norm_img * 255

cv2.imwrite('light.png', light_img)
```

## Python對象可视化

```python
import pickle

a = [[1],'2',3]

with open('a.pk','wb') as f:
    pickle.dump(a,f)
with open('a.pk','rb') as f:
    a_ = pickle.load(f)
print(a_) # [[1],'2',3]
```

## OpenCV基础图像处理

\*在OpenCV中RGB默认为BGR

# Image as a 2D Sampling of Signal

**Signal**: Function depends on some variables with physical meaning

**Image**: Sampling……

1D returns a vector while 2D returns a matrix.

视锥细胞：<img src="https://latex.codecogs.com/svg.image?{\color{Red}c}{\color{OliveGreen}on}{\color{Blue}es}" title="https://latex.codecogs.com/svg.image?{\color{Red}c}{\color{OliveGreen}on}{\color{Blue}es}" />
视感细胞：rods

<font size=3 color=D2691E>列表:</font>

<img src="https://latex.codecogs.com/svg.image?{\color{Green}&space;This\&space;is\&space;some\&space;text!}" title="https://latex.codecogs.com/svg.image?{\color{Green} This\ is\ some\ text!}" />

<font face="宋体">我是宋体字</font>

## Color Space HSV

**H**: Hue

**S**: Saturation

**V**: Value

# §2 Python图像进阶与全局图像变换



```python
import cv2
import mathplotlib
import numpy as np


```

# Transformation

## Global Transformation

1. Range transformation
2. Basic Point Operator

- Gamma Correction

	<img src="https://latex.codecogs.com/svg.image?s=c\cdot&space;r^\gamma" title="https://latex.codecogs.com/svg.image?s=c\cdot r^\gamma" />

- Negative Image

	<img src="https://latex.codecogs.com/svg.image?s=L-1-r" title="https://latex.codecogs.com/svg.image?s=L-1-r" />

- Log Transform

	<img src="https://latex.codecogs.com/svg.image?s=c\log(1&plus;r)" title="https://latex.codecogs.com/svg.image?s=c\log(1+r)" />

## Local Transformation

Filtering 来处理有用信息，去除无用信息。



# §3 图像变换

## 全局变换

```python
import cv2
import numpy as np

def hist_equ(gray):
    '''Conduct histogram equalization for scale image'''
    hist = np.histogram(gray, 256, [0, 256])
    norm_hist = hist[0] / (gray.shape[0] * gray.shape[1]) # 除以长宽以归一化
    integral = np.cumsum(norm_hist) # Cumulative Sum
    integral = (integral * 255.).astype(np.uint8)
    #now integral is s transformation function
    result = integral[gray] #Pixel-Wise mapping
    return result
img = cv2.imread('./example.jpg')
b, g, r = cv2.split(img) # 默认按最后一维拆分
nb, ng, nr = [hist_equ(gray) for gray in [b, g, r]]
result = np.stack([nb, ng, nr], -1)
cv2.imwrite('hist_equ.png', result)
```

问题是RGB空间三个分量分别做变换后白平衡会被破坏，所以用HSV空间更好

```python
import cv2
import numpy as np

def hist_equ(gray):
    '''Conduct histogram equalization for scale image'''
    hist = np.histogram(gray, 256, [0, 256])
    norm_hist = hist[0] / (gray.shape[0] * gray.shape[1]) # 除以长宽以归一化
    integral = np.cumsum(norm_hist) # Cumulative Sum
    integral = (integral * 255.).astype(np.uint8)
    #now integral is s transformation function
    result = integral[gray] #Pixel-Wise mapping
    return result

img = cv2.imread('./example.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
new_v = hist_equ(v)
new_hsv = np.stack([h, s, new_v], -1)
result = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('hist_equ_hsv.png', result)
```

## 局部变换

如何去除噪声？

- 图像具有平滑性、连续性
- 将每个像素值用周围像素值的平均值代替

# Detector and descriptor

## Properties of Ideal Local Feature

<font size=6 face="Times New Roman" color=7FA88>red</font>

Robusted to occlusion and clutter

1. Repeatability 
	- Given two images of the same object / scene, taken under different viewing angles.
	- The 
2. Distinctiveness / Informativeness
	- The intensity patterns underlying the detected features should show importance.
3. Locality
	- Features should be local, to reduce the probability of occlusion.
4. Quantity
	- The number of detected features should be sufficiently large.
5. Scale
6. Efficiency

## Edge detector

Edge: significant local changes in an image.
The gradient of an image:$\nabla f=[\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}]$

## Blob detector

## SIFT descriptor

Scale-Invariant Feature Transform

## HoG descriptor

Difference of Gaussian: DoG

## LBP descriptor

## Color descriptor

# §4 OpenCV人脸检测

<font size=5>综合以下三项给分：</font>

- 人脸检测性能
	- 评价指标：mean Average Precision (mAP)
	- 测试集：DarkFace + CVFace2022
- 方法创新性
- 扩展应用（综合考虑数量与实现难度）
	- UI
	- 更丰富的检测内容（如人脸landmark）
	- 低光视频增强与检测
	- ⋯⋯

<font size=5>最后提交<font color="red">代码</font>与<font color="red">项目报告</font></font>

安装Anaconda和PyTorch

# Motion & Optical Flow

## Three Factors in Computer Vision

\- Light
\- Object
\- Camera

## Varying Either of them causes motion

## Optical Flow（光流）

# §4 模式识别与机器学习简介

## 感知机

<img src="https://latex.codecogs.com/svg.image?\begin{array}{lcl}f(x)=\mathrm&space;{sgn}(\mathbf{w^T&space;x}&plus;b)\\y_i(\mathbf{w^T&space;x}_i&plus;b)\\b\to&space;b&plus;\eta&space;y_i\\\mathbf&space;w&space;\to&space;\mathbf&space;w&plus;\eta&space;y_i&space;\mathbf&space;x_i\end{array}" title="https://latex.codecogs.com/svg.image?\begin{array}{lcl}f(x)=\mathrm {sgn}(\mathbf{w^T x}+b)\\y_i(\mathbf{w^T x}_i+b)\\b\to b+\eta y_i\\\mathbf w \to \mathbf w+\eta y_i \mathbf x_i\end{array}" />

## 支持向量机（SVM）

Lagrange乘子法

## Bag of Feature



图像识别任务的基本步骤

1. 准备数据集（可能包括增强和预处理）
2. 提取特征（可能需要对特征进行向量编码）
3. 选择、训练分类器对特征向量进行分类
