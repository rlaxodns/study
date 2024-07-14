import numpy as np
from sklearn.preprocessing import StandardScaler

#1 데이터
data = np.array([[1,2,3,1],
                 [4,5,6,2],
                 [7,8,9,3],
                 [10,11,12,114],
                 [13,14,15,115]])

#1-1 평균
mean = np.mean(data, axis = 0)
print("평균", mean) #평균 [ 7.  8.  9. 47.]

#1-2 모집단 분산(/n : 나누기 n)
population_var = np.var(data, axis = 0) # ddof = 0 디폴트
print("분산", population_var) #[  18.   18.   18. 3038.]

#1-3 표본분산 (/(n-1) : 나누기(n-1))
var = np.var(data, axis = 0, ddof = 1)
print("표본분산", var) #표본분산 [  22.5   22.5   22.5 3797.5]

#1-4 표준편차
std = np.std(data, axis = 0, ddof = 1)
print("표본 표준편차", std) #표본 표준편차 [ 4.74341649  4.74341649  4.74341649 61.62385902]

#1-5 StandardScaler
std = StandardScaler()  # 모집단 분산으로 표준편차 계산된 표준정규분포
std_data =  std.fit_transform(data)
print("StandardScaler: \n", std_data)
"""
[[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
 [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
 [ 0.          0.          0.         -0.79828651]
 [ 0.70710678  0.70710678  0.70710678  1.21557264]
 [ 1.41421356  1.41421356  1.41421356  1.23371552]]
 """

# # 시각화
# import matplotlib.pyplot as plt
# plt.bar(std_data, )
# plt.show()