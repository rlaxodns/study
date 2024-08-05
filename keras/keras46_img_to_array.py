from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img  # 이미지 불러오는 라이브러리
from tensorflow.keras.preprocessing.image import img_to_array  # 불러온 이미지의 수치화
import matplotlib.pyplot as plt
import numpy as np



#경로
path = 'c://ai5//_data//image//me//me.jpg'

img = load_img(path,
               target_size=(100, 100))

print(img)       #<PIL.Image.Image image mode=RGB size=200x200 at 0x1E57F04A6A0>
print(type(img)) #<class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show() 

arr = img_to_array(img)
print(arr)
print(arr.shape)  # (200, 200, 3)
print(type(arr)) #<class 'numpy.ndarray'>

# 차원증가
img = np.expand_dims(arr, axis = 0)
print(img.shape) #(1, 100, 100, 3)


np.save("c://ai5//_data//image//me//me.npy", arr = img)