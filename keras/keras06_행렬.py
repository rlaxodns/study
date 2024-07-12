import numpy as np

x1 = np.array([1,2,3])
print("x1 :", x1.shape) #shape= 데이터의 모양을 보여줌 
                        # x1 : (3,)

x2 = np.array([[1,2,3]])
print("x2 :", x2.shape)

x3 =np.array([[1,2],[3,4]])
print("x3 :", x3.shape)

x4 = np.array([[1,2],[3,4],[5,6]])
print("x4:", x4.shape)

x5 = np.array([[[1,2],[3,4],[5,6]]])
print("x5:", x5.shape)

x6 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print("x6 :", x6.shape)

x7 = np.array([[[[1,2,3,4,5],[6,7,8,9,10]]]])
print("x7 :", x7.shape)

x8 = np.array([[1,2,3,],[4,5,6]])
print("x8 :", x8.shape)

x9 = np.array([[[[1,]]],[[[2]]]])
print("x :", x9.shape)

# x1 : (3,)        
# x2 : (1, 3)      
# x3 : (2, 2)      
# x4: (3, 2)       
# x5: (1, 3, 2)    
# x6 : (2, 2, 2)   
# x7 : (1, 1, 2, 5)
# x8 : (2, 3)      
# x : (2, 1, 1, 1)