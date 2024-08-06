# 46 copy

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
# np.save("c://ai5//_data//image//me//me.npy", arr = img)



###여기부터 증폭###
datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,        # 수평 뒤집기   <- 데이터 증폭 
    # vertical_flip=True,          # 수직 뒤집기 (상하좌우반전) <- 데이터 증폭
    width_shift_range=0.3,       # 평행이동  <- 데이터 증폭
    # height_shift_range=0.1,      # 평행이동 수직  <- 데이터 증폭
    rotation_range=15,            # 각도 조절 (정해진 각도만큼 이미지 회전)
    # zoom_range=1.2,              # 축소 또는 확대
    # shear_range=0.7,             # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환 (찌부시키기)
    fill_mode='nearest',         # 10% 이동 시 한쪽은 소실, 한쪽은 가까이에 있던 부분의 이미지로 채워짐
)

it = datagen.flow(img, batch_size=1,)   # flow는 수치화된 데이터를 가져오는 함수
print(it.next())

fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
# subplots: 연속적인 그림 출력

for i in range(5):
    batch = it.next()
    print(batch.shape)
    batch = batch.reshape(100,100,3)

    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()