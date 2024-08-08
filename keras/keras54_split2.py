import numpy as np
a = np.array([[1,2,3,4,5,6,7,8,9,10],
              [9,8,7,6,5,4,3,2,1,0]]).T
size = 5 # timesteps = 5

print(a.shape) #(10, 2)
print(a)

def split_x(dataset, size):
    aaa = []
    for i in range (len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)

x = bbb[:,:-1]
y = bbb[:, -1,0]
print(x,
    #    y
       )
       
print(x.shape, y.shape) #(6, 4, 2) (6,)

"""[[[ 6  4]
  [ 7  3]
  [ 8  2]
  [ 9  1]
  [10  0]]

 [[ 5  5]
  [ 6  4]
  [ 7  3]
  [ 8  2]
  [ 9  1]]

 [[ 4  6]
  [ 5  5]
  [ 6  4]
  [ 7  3]
  [ 8  2]]

 [[ 3  7]
  [ 4  6]
  [ 5  5]
  [ 6  4]
  [ 7  3]]

 [[ 2  8]
  [ 3  7]
  [ 4  6]
  [ 5  5]
  [ 6  4]]

 [[ 1  9]
  [ 2  8]
  [ 3  7]
  [ 4  6]
  [ 5  5]]]"""