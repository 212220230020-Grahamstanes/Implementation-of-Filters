# Implementation-of-Filters
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import the necessary modules.

### Step2
For performing smoothing operation on a image. 
- Average filter
```python
kernel=np.ones((11,11),np.float32)/121
image3=cv2.filter2D(image2,-1,kernel)
```
- Weighted average filter
```python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
```
- Gaussian Blur 
```python
gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
```
- Median filter
```python
median=cv2.medianBlur(image2,13)
```

### Step3
For performing sharpening on a image.
- Laplacian Kernel
```python
kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
```
- Laplacian Operator
```python
laplacian=cv2.Laplacian(image2,cv2.CV_64F)
```

### Step4
Display all the images with their respective filters.

## Program:
### Developed By   : Graham stanes
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

![image](https://user-images.githubusercontent.com/75235150/169489327-31977a04-37f1-449c-a63f-dc00d2807f6c.png)



ii) Using Weighted Averaging Filter

![image](https://user-images.githubusercontent.com/75235150/169489369-3b24d67d-2416-4ce2-bd80-77f2347506ec.png)


iii) Using Gaussian Filter

![image](https://user-images.githubusercontent.com/75235150/169489388-0e7ea751-c9c2-445d-a4e0-bd2ab2828fcb.png)


iv) Using Median Filter

![image](https://user-images.githubusercontent.com/75235150/169489421-aee8e90a-e85d-4e35-b5e6-753e47f2053d.png)

### 2. Sharpening Filters
i) Using Laplacian Kernal

![image](https://user-images.githubusercontent.com/75235150/169489445-a003c767-2cf7-4917-92db-ca61afda5969.png)


ii) Using Laplacian Operator

![image](https://user-images.githubusercontent.com/75235150/169489522-240112b1-c9b4-4768-a391-b60632416897.png))




## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
