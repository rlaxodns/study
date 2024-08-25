import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale = 2.0, size = 1000)
# print(data.shape) #(1000,)
# print(np.min(data), np.max(data)) #0.0003901236107508544, 14.642837980523108

log_data = np.log1p(data)

# 원본 데이터 히스토그램
plt.subplot(1,2,1)
plt.hist(data, bins = 50, color = 'blue', alpha = 0.5)
plt.title("Original")
# plt.show()

# 로그 변환 데이터 히스토그램
plt.subplot(1,2,2)
plt.hist(log_data, bins = 50, color = 'red', alpha = 0.5)
plt.title("Log Transformed")
plt.show()

# 원본으로 변환
data1 = np.expm1(log_data)
print(data1)