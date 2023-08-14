import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

import EvalFunctions as ev



def apply_and_evaluate_ensemble():
    """
    Apply and evaluate an ensemble of three models: RF, NN 3HL64N, and ELM.

    The function creates an ensemble of predictions from three different models:
    "RF (no log, yes pca)", "NN 3HL64N (no log, yes pca)", and "ELM (no log, yes pca)".
    
    It computes the ensemble prediction as the mean of predictions from the three models 
    and evaluates the ensemble performance on the test set using RMSE and MAE. Mispricing is 
    also calculated for the ensemble prediction relative to the actual 'Company Market Cap'. 
    Quintiles are added based on the mispricing, and the resulting dataframe is saved as a CSV.

    The function prints out the RMSE and MAE for the test set in Billion USD. Additionally, 
    these metrics, along with a timestamp and the model name, are appended to a file named 'valuation_accuracy'.

    Returns:
        None

    Side-effects:
        - Modifies or creates CSV files in the directories 'model_evaluations_1/' and 'model_evaluations_4/'.
        - Prints the ensemble model's RMSE and MAE on the console.

    Raises:
        AssertionError: If the number of quintile groups in 'return_summary' is not 5.

    Notes:
        - Assumes that 'ev.get_ev1_df_list()' is a function that retrieves evaluation dataframes 
          for the ensemble models.
        - Assumes that 'ev.add_quintiles()' is a function that adds quintile information to the dataframe.
    """
    
    ensemble_models = ["RF (no log, yes pca)", "NN 3HL64N (no log, yes pca)", "ELM (no log, yes pca)"]

    ev1_ensemble_dfs = ev.get_ev1_df_list(ensemble_models)
    
    merged = pd.merge(ev1_ensemble_dfs[0][['Date', 'tic', 'Company Market Cap', 'pred']],
                  ev1_ensemble_dfs[1][['Date', 'tic', 'pred']], on=['Date', 'tic'], how='left')
         
    merged = pd.merge(merged, ev1_ensemble_dfs[2][['Date', 'tic', 'pred']],
                  on=['Date', 'tic'], how='left')
    
    
    merged['ensemble_pred'] = merged[['pred_x', 'pred_y', 'pred']].mean(axis=1)
    merged.drop(columns=['pred_x', 'pred_y', 'pred'], inplace=True)
    
    
    testset_eval = merged.copy()

    rmse = mean_squared_error(testset_eval['Company Market Cap'], testset_eval['ensemble_pred'], squared=False)
    mae = mean_absolute_error(testset_eval['Company Market Cap'], testset_eval['ensemble_pred'])
    
    
    
    testset_eval['Mispricing'] = (testset_eval['ensemble_pred'] - testset_eval['Company Market Cap']) / testset_eval['Company Market Cap']
    
    ev.add_quintiles(testset_eval, name='Quintile')
    
    next_return_df = pd.read_csv('data/next_returns.csv', index_col=0, parse_dates=[2])
    next_return_df['Date'] = next_return_df['Date'].dt.to_period('M')
    testset_eval = pd.merge(testset_eval, next_return_df, on=['Date', 'tic'], how='left')
    return_summary = testset_eval[['Quintile', 'Mispricing', 'Next 1M Return', 'Next 6M Return', 'Next 12M Return']].groupby('Quintile').mean()
    
    assert return_summary.shape[0] == 5, f"Error: Expected 5 groups but got {return_summary.shape[0]} groups."
    
    name="Ensemble"
    
    testset_eval.to_csv(f'model_evaluations_1/{name}.csv')
    
    print("Test Set (2019-2023)")
    print("Root Mean Squared Error:", round(rmse/1000000000, 2), "Billion USD")
    print("Mean Absolute Error:", round(mae/1000000000, 2), "Billion USD")
    
    
    with open('model_evaluations_4/valuation_accuracy', 'a') as file:
        file.write('\n' + 80*'#')
        file.write(f"\n\nTIMESTAMP: {datetime.now()}\n")
        file.write(f"MODEL: {name}\n\n")
        file.write("Test Set (2019-2023)")
        file.write(f"\nRoot Mean Squared Error: {(rmse/1000000000):.2f} Billion USD")
        file.write(f"\nMean Absolute Error: {(mae/1000000000):.2f} Billion USD\n")