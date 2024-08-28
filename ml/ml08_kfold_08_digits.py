from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#데이터
x, y = load_digits(return_X_y=True)
print(x.shape, y.shape) #(1797, 64) (1797,)
print(pd.value_counts(y, sort=False))

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
score = cross_val_score(model, x, y, 
                        cv = kfold)

print("score", score, "평균score", np.mean(score))
# score [0.98054475 0.98832685 0.9922179  0.98832685 0.9922179  0.98828125
#  0.99609375] 평균score 0.989429891606448

# score [1.         0.98054475 0.98054475 0.99610895 0.9766537  0.984375
#  0.9921875 ] 평균score 0.9872020914396887