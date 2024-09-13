from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from src.common.evaluation import evaluate_model


class KNNModel:
    def __init__(self, X_train, y_train, X_test, y_test, cv, param_grid, random_state=17):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.param_grid = param_grid
        self.model = KNeighborsClassifier()
        self.random_state = random_state
        self.best_model = None
        self.cv_results = None
        self.best_params = None

    def train(self):
        grid_search = GridSearchCV(self.model, param_grid=self.param_grid, cv=self.cv, n_jobs=-1, return_train_score=True, verbose=2)

        grid_search.fit(self.X_train, self.y_train)

        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_: .3f}')

        # log the best model and evaluate
        best_model = grid_search.best_estimator_
        self.best_model = best_model
        self.cv_results = grid_search.cv_results_
        self.best_params = grid_search.best_params_

    def evaluate(self):
        y_pred = self.best_model.predict(self.X_test)
        evaluation_metrics = evaluate_model(self.y_test, y_pred)
        return evaluation_metrics