import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, learning_curve
from src.common.evaluation import evaluate_model
from sklearn.model_selection import train_test_split

class Classifier:
    def __init__(self, model, X_train, y_train, X_test, y_test, param_grid, cv, random_state=17):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        self.param_grid = param_grid
        self.random_state = random_state
        self.model_type = model
        self.best_model = None
        self.cv_results = None
        self.best_params = None

        if self.model_type == 'knn':
            self.model = KNeighborsClassifier()
        elif self.model_type == 'nn':
            self.model = MLPClassifier(random_state=self.random_state, \
                                       early_stopping=True)
        elif self.model_type == 'svm':
            self.model = SVC(random_state=self.random_state)
        elif self.model_type == 'boosting':
            self.model = GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError('Invalid model. Choose from knn, nn, or svm')

    def train(self):
        grid_search = GridSearchCV(self.model, param_grid=self.param_grid, cv=self.cv, n_jobs=-3, return_train_score=True, verbose=3)

        grid_search.fit(self.X_train, self.y_train)
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best score: {grid_search.best_score_:.3f}')

        self.best_model = grid_search.best_estimator_
        self.cv_results = grid_search.cv_results_
        self.best_params = grid_search.best_params_

    def evaluation(self):
        y_pred = self.best_model.predict(self.X_test)
        evaluation_metrics = evaluate_model(self.y_test, y_pred)
        return evaluation_metrics

    def create_learning_curve_data(self):
        train_sizes, train_scores, val_scores, fit_times, _ \
            = learning_curve(self.best_model, self.X_train, self.y_train, \
                             train_sizes=[0.2, 0.4, 0.6, 0.8, 1], \
                             cv=self.cv, n_jobs=-3, random_state=self.random_state, \
                             verbose=3, return_times=True)
        learning_curve_data = {
            'train_sizes': train_sizes.tolist(),
            'train_scores': train_scores.tolist(),
            'val_scores': val_scores.tolist(),
            'fit_times': fit_times.tolist()
        }
        return learning_curve_data

    def nn_learning_curve_data(self, epochs=50):
        # create a new model with the best parameters
        model = MLPClassifier(random_state=self.random_state, **self.best_params)
        train_scores = []
        val_scores = []

        # Split training data into train and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=self.random_state
        )

        for i in range(1, epochs + 1):
            model.partial_fit(X_train_split, y_train_split, classes=[0, 1])
            train_score = model.score(X_train_split, y_train_split)
            val_score = model.score(X_val, y_val)
            train_scores.append(train_score)
            val_scores.append(val_score)

            if i % 10 == 0:
                print(f'Epoch: {i}, Train score: {train_score:.3f}, Val score: {val_score:.3f}')

        learning_curve_data = {
            'train_scores': train_scores,
            'val_scores': val_scores,
            'epochs': list(range(1, epochs + 1))
        }
        return learning_curve_data



