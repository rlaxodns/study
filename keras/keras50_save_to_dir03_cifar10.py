from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True, 
    # width_shift_range=0.5,
    # height_shift_range=0.5, 
    # rotation_range=25, 
    # # zoom_range=0.1,
    # # shear_range = 0.5,
    fill_mode='nearest'
)

# print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)

augment_size = 10000

randidx = np.random.randint(x_train.shape[0], size = augment_size)
# print(randidx)
# print(np.min(randidx), np.max(randidx)) #3 49999

x_aug = x_train[randidx].copy()
y_aug = y_train[randidx].copy()

print(x_aug.shape) #(50000, 32, 32, 3)

x_aug = train_datagen.flow(
    x_aug,
    y_aug,
    batch_size=augment_size,
    shuffle=False,
    save_to_dir="./_data/_save_img/03_cifar10"
).next()[0]
# print(x_aug.shape)
# print(x_train.shape)

