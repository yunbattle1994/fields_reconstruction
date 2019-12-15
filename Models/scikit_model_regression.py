####3.1决策树回归####
def decision_tree_regressor(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeRegressor()
    model.fit(train_x, train_y)
    return model
####3.2线性回归####
def linear_regressor(train_x, train_y):
    from sklearn import linear_model
    model = linear_model.LinearRegression(fit_intercept=False, n_jobs=10)
    model.fit(train_x, train_y)
    return model
####3.3SVM回归####
def support_vector_regressor(train_x, train_y):
    from sklearn import svm

    from sklearn import grid_search

    model = svm.SVR(degree=10, C=100, kernel="poly", gamma='auto', epsilon=1e-7)
    parameters = {"kernel": ("linear", "rbf", "poly"), "C": range(1, 100), 'degree': range(1, 10)}
    # model = grid_search.GridSearchCV(svr, parameters)
    model.fit(train_x, train_y)
    # print(model.best_params_)
    return model
####3.4KNN回归####
def kneighbors_regressor(train_x, train_y):
    from sklearn import neighbors
    model = neighbors.KNeighborsRegressor()
    model.fit(train_x, train_y)
    return model
####3.5随机森林回归####
def random_forest_regressor(train_x, train_y):
    from sklearn import ensemble
    model = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
    model.fit(train_x, train_y)
    return model
####3.6Adaboost回归####
def adaboost_regressor(train_x, train_y):
    from sklearn import ensemble
    model = ensemble.AdaBoostRegressor(n_estimators=100)#这里使用50个决策树
    model.fit(train_x, train_y)
    return model
####3.7GBRT回归####
def gradient_boosting_regressor(train_x, train_y):
    from sklearn import ensemble
    model = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
    model.fit(train_x, train_y)
    return model
####3.8Bagging回归####
def bagging_regressor(train_x, train_y):
    from sklearn.ensemble import BaggingRegressor
    model = BaggingRegressor()
    model.fit(train_x, train_y)
    return model

####3.9ExtraTree极端随机树回归####
def extra_tree_regressor(train_x, train_y):
    from sklearn.tree import ExtraTreeRegressor
    model = ExtraTreeRegressor()
    model.fit(train_x, train_y)
    return model

def xgboost_regressor(train_x, train_y):
    import xgboost as xgb
    model = xgb.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.1,
                             reg_lambda=1, reg_alpha=0, gamma=0.00001, n_jobs=30)
    model.fit(train_x, train_y)
    return model

def gauss_regressor(train_x, train_y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C  # REF就是高斯核函数
    import sklearn.gaussian_process as gp

    # kernel = gp.kernels.Matern()

    kernel = C(0.1, (1e-5, 1e5)) * RBF(0.1, (1e-6, 1e6))

    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)

    model.fit(train_x, train_y)

    return model