# 실제의 데이터에서는 x데이터만 존재하기 때문에 y데이터와 timesteps와 feature를 나누는 과정이 필요하다.

import numpy as np
a = np.array(range(1,11))
size = 5 # timesteps = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
# print(bbb)
# print(bbb.shape)

x = bbb[:,:-1]
y = bbb[:, -1]
print(x,y)
print(x.shape, y.shape) #(6, 4) (6,)