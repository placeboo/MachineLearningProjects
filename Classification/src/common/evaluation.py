from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.common.utils import load_model, load_metrics, plot_learning_curve, save_plot, save_model, save_metrics, load_cv_results, format_cv_results, plot_complexity_curve, plot_training_time


def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using accuracy, precision, recall, and f1-score
    :param y_true: the numpy array of the true target values
    :param y_pred: the numpy array of the predicted target values
    :return: a dictionary containing the evaluation metrics and show the confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'f1-score: {f1:.3f}')

    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def summarize_model(input_dir, output_dir, model_name, dataset_name, y_label='Error Rate'):
    # laod the best model
    best_model = load_model(input_dir, model_name, dataset_name)
    metrics = load_metrics(input_dir, model_name, dataset_name)
    print(f'the best model params: {best_model.get_params()}')
    print(f'the metrics of testing dataset: {metrics}')

    # learning curve
    lc_data = load_metrics(input_dir, model_name, f'{dataset_name}_lc')
    lc_plot = plot_learning_curve(lc_data, model_name, y_label=y_label)
    # save
    save_plot(lc_plot, output_dir, model_name, 'lc', dataset_name)
    return best_model, metrics

def summarize_complexity_curve(cv_results_df, output_dir, model_name, dataset_name, param_name,ylabel='F1-Score'):

    # print the row with the highest score
    print(cv_results_df.loc[cv_results_df['mean_test_score'].idxmax()])

    # plot the complexity curve
    cc_plt, ax = plot_complexity_curve(cv_results_df, param_name, 'mean_train_score', 'mean_test_score', f'Model Complexity Curve: {param_name}', ylabel=ylabel)

    # save the plot
    save_plot(cc_plt, output_dir, model_name, f'cc_{param_name}', dataset_name)
    return cc_plt, ax

