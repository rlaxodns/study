import pandas as pd
import numpy as np

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])
print(data.shape) #(4, 5)
data = data.transpose()
data.columns = ['x1', 'x2', 'x3', 'x4']

# print(data)
    #  x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# print(data.shape) #(5, 4)

# print(data.isna().sum())
# print(data.isnull())
# print(data.info())
# # x1    1
# x2    2
# x3    0
# x4    3

#1. 결측치 삭제
# print(data.dropna())
#     x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

# print(data.dropna(axis=0)) # 행삭제
    # x1   x2   x3   x4
# 3  8.0  8.0  8.0  8.0

# print(data.dropna(axis=1)) # 열삭제
#      x3
# 0   2.0
# 1   4.0
# 2   6.0
# 3   8.0
# 4  10.0

#2-1 특정값 - 평균
means = data.mean()
print(means)

data2 = data.fillna(means)
print(data2)

#2-2 특정값 - 중위값
med = data.median()
print(med)

data3 = data.fillna(med)
print(data3)

#2-3 특정값 - 0 또는 임의의 값
data4 = data.fillna(0)
print(data4)

data4_2 = data.fillna(777)
print(data4_2)

#2-4 특정값 - ffill (통상 마지막값이나 맨 앞에서 많이 사용)
# data5 = data.ffill()
data5 = data.fillna(method = 'ffill')
# data5_2 = data.bfill()
data5_2 = data.fillna(method = 'bfill')
print(data5)
print(data5_2)

### 특정 컬럼만 ###
means = data['x1'].mean()
print(means) # 6.5

meds = data['x4'].median()
print(meds) # 6.0

data['x1'] = data['x1'].fillna(means)
data['x4'] = data['x4'].fillna(meds)
data['x2'] = data['x2'].ffill()
print(data)