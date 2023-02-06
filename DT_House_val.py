import pandas as pd
import numpy as np
import statistics as st
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("training_data_v1.csv")

x = data.drop(["LOGVALUE"], axis= 1)
y = data["LOGVALUE"]

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=0, test_size=0.3)


def opt_features(): 
    kb = SelectKBest(score_func=f_regression, k="all")
    fit = kb.fit(x,y)
    np.set_printoptions(precision=2)
    scores = fit.scores_
    print(scores)

    ranked_scores = []
    while len(scores) > 0 :
        ranked_scores += [max(scores)]
        scores = scores[scores != max(scores)]

    np.set_printoptions(precision=2)
    print(ranked_scores)
    # 0, 
# The output of this feature selection is that Baths, bedrooms, unit square foot, rooms and region are the most important features



def dt_reg_all():
    model = DecisionTreeRegressor(random_state=0)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)

    mse = mean_squared_error(ytest, pred)
    rsq = r2_score(ytest, pred)


    cv = KFold(10)
    cv_mse = cross_val_score(model, x, y, cv=cv, scoring="neg_mean_squared_error")
    cv_score = st.mean(abs(cv_mse))
    print(f'The MSE using all features is {mse} and the R^2 score is {rsq}, and the cv_mse is {cv_score}')


def dt_reg_five():
    x_5 = data.drop(["LOGVALUE", "BUILT", "LOT", "KITCHEN", "FLOORS", "LAUNDY", "RECRM", "METRO", "METRO3"], axis= 1)
    x5_train, x5_test, y5_train, y5_test = train_test_split(x_5, y, test_size=0.3, random_state=0)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x5_train, y5_train)
    pred = model.predict(x5_test)
    mse = mean_squared_error(y5_test, pred)
    rsq = r2_score(y5_test, pred)

    cv = KFold(10)
    cv_mse = cross_val_score(model, x_5, y, cv=cv, scoring="neg_mean_squared_error")
    cv_score = st.mean(abs(cv_mse))
    print(f'The MSE using select 5 features is {mse} and the R^2 score is {rsq}, and the cv_mse is {cv_score}')
    pass

def dt_reg_four():
    x_4 = data.drop(["LOGVALUE", "BUILT", "LOT", "KITCHEN", "FLOORS", "LAUNDY", "RECRM", "METRO", "METRO3", "UNITSF"], axis= 1)
    x4_train, x4_test, y4_train, y4_test = train_test_split(x_4, y, test_size=0.3, random_state=0)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x4_train, y4_train)
    pred = model.predict(x4_test)
    mse = mean_squared_error(y4_test, pred)
    rsq = r2_score(y4_test, pred)
    

    cv = KFold(10)
    cv_mse = cross_val_score(model, x_4, y, cv=cv, scoring="neg_mean_squared_error")
    cv_score = st.mean(abs(cv_mse))
    print(f'The MSE using select 4 features is {mse} and the R^2 score is {rsq}, and the cv_mse is {cv_score}')
    
    importance = model.feature_importances_
    print(importance)
    
    plt.bar(["BATHS", "BEDRMS", "ROOMS", "REGION"], importance)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()
    pass

def dt_reg_three():
    x_3 = data.drop(["LOGVALUE", "BUILT", "LOT", "KITCHEN", "FLOORS", "LAUNDY", "RECRM", "METRO", "METRO3", "UNITSF", "REGION"], axis= 1)
    x3_train, x3_test, y3_train, y3_test = train_test_split(x_3, y, test_size=0.3, random_state=0)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x3_train, y3_train)
    pred = model.predict(x3_test)
    mse = mean_squared_error(y3_test, pred)
    rsq = r2_score(y3_test, pred)

    cv = KFold(10)
    cv_mse = cross_val_score(model, x_3, y, cv=cv, scoring="neg_mean_squared_error")
    cv_score = st.mean(abs(cv_mse))
    print(f'The MSE using select 3 features is {mse} and the R^2 score is {rsq}, and the cv_mse is {cv_score}')  


def pred_comp():
    x_pc = data.drop(["LOGVALUE", "BUILT", "LOT", "KITCHEN", "FLOORS", "LAUNDY", "RECRM", "METRO", "METRO3", "UNITSF"], axis= 1)
    target = pd.read_csv("test_data_v1.csv")
    targetx = target.drop(["LOGVALUE", "BUILT", "LOT", "KITCHEN", "FLOORS", "LAUNDY", "RECRM", "METRO", "METRO3", "UNITSF"], axis= 1)
    targety = target["LOGVALUE"]
    model = DecisionTreeRegressor(random_state=0)
    model.fit(x_pc, y)
    pred = model.predict(targetx)
    mse = mean_squared_error(targety, pred)
    rsq = r2_score(targety, pred)
    print(f"The MSE of the model using the validation data is {mse}, whereas the R^2 value is {rsq}")



if __name__ == "__main__":
    # opt_features()
    # dt_reg_all()
    # dt_reg_five()
    # dt_reg_four()
    # dt_reg_three()
    pred_comp()