from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
d1 = fetch_california_housing()
d2 = load_diabetes()

d = [d1, d2]
Random_state = 8888
for i in d:
    x = i.data
    y = i.target
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size= 0.2, 
                                                        random_state=Random_state, 
                                                        )
    
    model1 = DecisionTreeRegressor(random_state=Random_state)
    model2 = RandomForestRegressor(random_state=Random_state)
    model3 = GradientBoostingRegressor(random_state=Random_state)
    model4 = XGBRegressor(random_state = Random_state)

    models = [model1, model2, model3, model4]

    
    for model in models:
        model.fit(x_train, y_train)
        print('========================', model.__class__.__name__, 'Random_state = ',Random_state, "======================")
        print('acc', model.score(x_test, y_test))
        print(model.feature_importances_)