import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

import BGLR
import EvalFunctions as ev

def apply_LR(Xtrn_full_minmax, Ytrn_full, Xtst_minmax, Ytst, log_transformed=False, save_eval_df=False, name="LR"):
    """
    Applies Linear Regression (LR) model to the provided datasets, evaluates the model, and optionally saves the evaluation results.

    Args:
        Xtrn_full_minmax (numpy.ndarray): The normalized training data.
        Ytrn_full (numpy.ndarray): The training labels.
        Xtst_minmax (numpy.ndarray): The normalized test data.
        Ytst (numpy.ndarray): The test labels.
        log_transformed (bool, optional): A flag indicating whether the response variable has been log-transformed. Default is False.
        save_eval_df (bool, optional): A flag indicating whether to save the evaluation DataFrame. Default is False.
        name (str, optional): The name of the model, used when saving the evaluation results. Default is "LR".

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation results.
    """
    
    lr, mse, r2 = BGLR.apply_lin_regression(Xtrn_full_minmax, Ytrn_full)
    
    print("R-squared (Train Set):", round(r2, 4), "\n")
    
    ev2_df = ev.evaluate_quintile_returns(lr,
                                          Xtst_minmax,
                                          Ytst,
                                          log_transformed=log_transformed,
                                          save_eval_df=save_eval_df,
                                          name=name)
    
    if save_eval_df:
        ev2_df.to_csv(f'model_evaluations_2/{name}.csv')
    
    return ev2_df