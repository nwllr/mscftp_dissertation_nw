from sklearn.ensemble import RandomForestRegressor

import DimReduction
import EvalFunctions as ev


def apply_RF(Xtrn_full, Ytrn_full, Xtst, Ytst, log_transformed=False, pca=False,
             save_eval_df=False, name="RF"):
    """
    Applies a Random Forest (RF) model to the provided datasets, evaluates the model, and optionally saves the evaluation results.

    Args:
        Xtrn_full (numpy.ndarray): The training data.
        Ytrn_full (numpy.ndarray): The training labels.
        Xtst (numpy.ndarray): The test data.
        Ytst (numpy.ndarray): The test labels.
        log_transformed (bool, optional): A flag indicating whether the response variable has been log-transformed. Default is False.
        pca (bool, optional): A flag indicating whether to apply Principal Component Analysis (PCA) on the input data. Default is False.
        save_eval_df (bool, optional): A flag indicating whether to save the evaluation DataFrame. Default is False.
        name (str, optional): The name of the model, used when saving the evaluation results. Default is "RF".

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation results.
    """

    if pca:
        Xtrn_full, Xtst, nPCs = DimReduction.apply_pca(
            Xtrn_full=Xtrn_full, Xtst=Xtst, val_set=False)
        
        
    
    rf = RandomForestRegressor()
    rf.fit(Xtrn_full, Ytrn_full)
    
    ev2_df = ev.evaluate_quintile_returns(rf,
                                          Xtst,
                                          Ytst,
                                          log_transformed=log_transformed,
                                          save_eval_df=save_eval_df,
                                          name=name)
    
    if save_eval_df:
        ev2_df.to_csv(f'model_evaluations_2/{name}.csv')
        
        
    return ev2_df


