import pandas as pd
import numpy as np
import statistics as st
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("training_data_v1.csv")
# Loads the training dataset

x = data.drop(["LOGVALUE"], axis= 1)
y = data["LOGVALUE"]
# splits the dataset into features (y) and target variables (x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=0, test_size=0.3)
# splits the feature and target data into training and testing data using a randomization algorithm built into train test split.
# 30% of the data is reserved as validation data


def opt_features():
    # This function uses Select K Best from SKLearn to determine f scores for the features of the datset
    # the f scores are fiest displayed in order of feature appearance, which is then sorted from most to least
    # The output of this feature selection is that Baths, bedrooms, unit square foot, rooms and region are the most important feature
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



def dt_reg_all():
    # This function uses Decision Trees to predict housing prices using all of the features in the dataset.
    # The output of this function is the MSE, the R2, and the cross-validation MSE using KFold where K = 10 
    model = DecisionTreeRegressor(random_state=0)
    model.fit(xtrain, ytrain)
    pred = model.predict(xtest)

    mse = mean_squared_error(ytest, pred)
    rsq = r2_score(ytest, pred)


    cv = KFold(10)
    cv_mse = cross_val_score(model, x, y, cv=cv, scoring="neg_mean_squared_error")
    cv_score = st.mean(abs(cv_mse))
    print(f'The MSE using all features is {mse} and the R^2 score is {rsq}, and the cv_mse is {cv_score}')
    # The MSE using all features is 1.6103096485531745 and the R^2 score is -0.24645350250248277, and the cv_mse is 1.6496978963754971


def dt_reg_five():
    # This function uses Decision Trees to predict housing prices using the features that had the 5 highest
    # f scores as determined by the opt_features function.
    # The output of this function is the MSE, the R2, and the cross-validation MSE using KFold where K = 10 
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
    # The MSE using select 5 features is 1.366365958448709 and the R^2 score is -0.057629901266977424, and the cv_mse is 1.2619657750395326

def dt_reg_four():
    # This function uses Decision Trees to predict housing prices using the features that had the 4 highest
    # f scores as determined by the opt_features function.
    # The output of this function is the MSE, the R2, and the cross-validation MSE using KFold where K = 10
    # this model was found to be the most accurate model, as it had the lowest MSE score, thus the importance
    # of the features in this model was also calculated and displayed using a bar graph 
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
    print(f'The feature importance scores {importance} for the BATHS, BEDRMS, ROOMS, and REGION features respectively')
    
    plt.bar(["BATHS", "BEDRMS", "ROOMS", "REGION"], importance)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.show()
    # The MSE using select 4 features is 1.011711963175417 and the R^2 score is 0.21688855236215776, and the cv_mse is 0.921828599330189

def dt_reg_three():
    # This function uses Decision Trees to predict housing prices using the features that had the 5 highest
    # f scores as determined by the opt_features function.
    # The output of this function is the MSE, the R2, and the cross-validation MSE using KFold where K = 10 
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
    # The MSE using select 3 features is 1.1127378635276264 and the R^2 score is 0.13868987333752547, and the cv_mse is 1.009166622777461 


def pred_comp():
    # This function uses Decision Trees to predict housing prices using the the best (4-feature) model.
    # The output of this function is the MSE, the R2, and the cross-validation MSE using KFold where K = 10
    # Unlike in model selection, this function predicticts housing prices using a data set that was collected after the training of the model.
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
    # The MSE of the model using the validation data is 0.7669263626672429, whereas the R^2 value is 0.28146667711547046


if __name__ == "__main__":
    opt_features()
    dt_reg_all()
    dt_reg_five()
    dt_reg_four()
    dt_reg_three()
    pred_comp()