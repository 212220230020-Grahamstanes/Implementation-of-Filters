### EX NO : 06
### DATE  : 12.05.2022
# <p align="center">Implementation-of-Filters</p>

## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## ALGORITHM:
### Step 1:
Import the necessary modules. 
### Step 2:
Perform smoothing operation on a image. 
- Average filter
- Weighted average filter
- Gaussian Blur 
- Median filter
### Step 3:
Perform sharpening on a image.
- Laplacian Kernel
- Laplacian Operator
### Step 4:
Display all the images with their respective filters.

<br/>
<br/>
<br/>

## Program:
### Developed By   : Graham Stanes
### Register Number: 212220230020
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("simp.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
```

### 1. Smoothing Filters

i) Using Averaging Filter
```Python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()
```

ii) Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()
```

iii) Using Gaussian Filter
```Python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```

iv) Using Median Filter
```Python
median=cv2.medianBlur(image2,13)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Median Blur")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Kernal
```Python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()
```

ii) Using Laplacian Operator
```Python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()
```

## OUTPUT:
### 1. Smoothing Filters
i) Using Averaging Filter

![1](https://user-images.githubusercontent.com/75235488/168803429-47fa8bc9-6994-42c8-967c-c1a88f69bd60.png)


ii) Using Weighted Averaging Filter

![2](https://user-images.githubusercontent.com/75235488/168803449-6a23cdbd-7050-4f69-8f0c-d9da28fcba10.png)

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

iii) Using Gaussian Filter

![3](https://user-images.githubusercontent.com/75235488/168803461-301e8a35-5742-495b-880e-41ca09192e30.png)


iv) Using Median Filter

![4](https://user-images.githubusercontent.com/75235488/168803476-213cf819-d57d-4d76-8a5c-f3f81cfe9588.png)


### 2. Sharpening Filters
i) Using Laplacian Kernal

![5](https://user-images.githubusercontent.com/75235488/168803506-56181e12-7725-476c-a186-69c9f3223a42.png)

<br/>
<br/><br/>
<br/><br/>
<br/><br/>
<br/>

ii) Using Laplacian Operator

![6](https://user-images.githubusercontent.com/75235488/168803521-679b16b7-6324-4e67-a646-f871fe021960.png)




## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
