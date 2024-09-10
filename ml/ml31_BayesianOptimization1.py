param_bounds = {'x1':(-1, 5), # 0
                'x2': (3, 7)} # 2

def y_function(x1, x2):
    return -x1**2 - (x2-2)**2 + 10
# 우리는 y의 최대값이 10이 나오게 하려면 x1 = 0, x2=2일 경우 y의 최대값인 걸 알 수 있으나, 
# 머신이 이러한 최적화를 할 수 있도록 베이지안 최적화를 활용한다.

from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f = y_function,
    pbounds = param_bounds,
    random_state=333,
)

optimizer.maximize(init_points=5,
                   n_iter=20)

print(optimizer.max)