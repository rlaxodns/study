from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )

# ohe = OneHotEncoder(sparse=False)
# y = y.reshape(-1, 1)
# y = ohe.fit_transform(y)

kfold = StratifiedKFold(
    n_splits = 7, 
    shuffle=True, 
    random_state=777
)

# 모델 구성
model = SVC()

# 평가 및 예측
score = cross_val_score(model, x_train, y_train, 
                        cv = kfold)

print("score", score, "평균score", np.mean(score))

y_pre = cross_val_predict(model, x_test, y_test, 
                          cv = kfold)
acc = accuracy_score(y_test, y_pre)
print(acc)
# score [0.98054475 0.98832685 0.9922179  0.98832685 0.9922179  0.98828125
#  0.99609375] 평균score 0.989429891606448

# score [1.         0.98054475 0.98054475 0.99610895 0.9766537  0.984375
#  0.9921875 ] 평균score 0.9872020914396887

# score [0.99029126 0.98058252 0.99512195 0.97560976 0.9902439  0.9902439
#  0.9902439 ] 평균score 0.9874767430059875
# 0.9555555555555556