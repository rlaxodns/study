import autokeras as ak
import tensorflow as tf
print(ak.__version__)
print(tf.__version__)

import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # 로그값을 줄이는 로그레벨

#1. 데이터
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

#2. 모델
model = ak.ImageClassifier(
    overwrite=False,
    max_trials=3,

)

#3. 컴파일 훈련
st = time.time
model.fit(x_train, y_train,
          epochs=100,
          validation_split=0.15)
et = time.time

####최적 출력 모델####
best_model = model.export_model()
print(best_model.summary())


###최적의 모델 저장###
path = 'C:\\ai5\\_data\\_save\\autokeras\\'
best_model.save(path+"keras70_autokeras1.h5")

#4. 평가 예측
y_pre = model.predict(x_test)
result = model.evaluate(x_test, y_test)
print("model의 결과", result)

y_pre2 = best_model.predict(x_test)
result2 = model.evaluate(x_test, y_test)
print("model의 결과", result2)
# print("걸린시간", et-st)
