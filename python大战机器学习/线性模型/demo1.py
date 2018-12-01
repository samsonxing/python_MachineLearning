import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis,model_selection

def load_data():
    diabetes = datasets.load_diabetes()
    # 返回的是训练样本集、测试样本集、训练样本集和测试样本集的比例
    return model_selection.train_test_split(diabetes.data,diabetes.target,test_size= 0.25,random_state = 0)

# 线性回归模型
def test_LinearRegression(*data):
    X_train,X_test,y_train,y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    print('Coefficient:%s, intercept %.2f'%(regr.coef_,regr.intercept_))
    print("Residual sum of square: %2f"% np.mean((regr.predict(X_test)- y_test)**2))
    print('Score: %.2f'% regr.score(X_test,y_test))

X_train,X_test,y_train,y_test = load_data()
test_LinearRegression(X_train,X_test,y_train,y_test)

def test_Ridge(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train,y_train)
    print('Coefficient:%s, intercept %.2f' % (regr.coef_, regr.intercept_))
    print("Residual sum of square: %2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, y_test))

X_train,X_test,y_train,y_test = load_data()
test_Ridge(X_train,X_test,y_train,y_test)
