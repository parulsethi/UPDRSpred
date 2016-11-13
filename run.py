import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, learning_curve
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from feature_selection import import_data

np.random.seed(1)

def plot_feature_imp(model, X_train, feature_title):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    f_t = np.array(feature_title)
    plt.figure()
    plt.ylabel("Feature importances", labelpad=15, fontsize=14, fontweight='bold')
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="b", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), f_t[indices], rotation='vertical')
    plt.xlim([-1, X_train.shape[1]])
    plt.subplots_adjust(bottom=0.22)
    plt.show()

def plot_learning_curve(model, X, y):
    plt.figure()
    train_sizes, train_scores, test_scores = \
        learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1, 10),
                       scoring="r2", cv=4)
    plt.plot(train_sizes, test_scores.mean(1), 'o-', color="g",
             label="Test data")
    plt.plot(train_sizes, train_scores.mean(1), 'o-', color="r",
             label="Train data")
    plt.xlabel("Train size")
    plt.ylabel("R^2 score")
    plt.title('Learning curves')
    plt.legend(loc="best")
    plt.show()

f_data = 'data.json'
n_features = 18
X, y, f_titles = import_data(f_data, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y)

p_grid = {"n_estimators": [5, 10, 20, 30],
          "max_features": [8, 10, 12, 15, 18],
          "criterion": ["mse", "mae"],
          "max_depth": [3, None],
          "bootstrap": [True, False]
          }

cv = KFold(n_splits=4, shuffle=True)
reg = GridSearchCV(estimator=RandomForestRegressor(), param_grid=p_grid, cv=cv)

reg.fit(X_train, y_train)

# plot_feature_imp(reg.best_estimator_, X_train, f_titles)
# plot_learning_curve(reg, X_train, y_train)

y_pred = reg.predict(X_test)

print(reg.best_estimator_)
print(reg.best_params_)
print(reg.score(X_test, y_test))

