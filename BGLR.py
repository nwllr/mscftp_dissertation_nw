import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


import EvalFunctions as ev

def apply_lin_regression(X, y):
    """
    Applies a linear regression model on the provided features and target.
    
    Args:
        X (pd.DataFrame or np.ndarray): Features for the linear regression.
        y (pd.Series or np.ndarray): Target variable.

    Returns:
        tuple:
            - lr (LinearRegression model): Fitted linear regression model.
            - mse (float): Mean squared error of the predictions.
            - r2 (float): R^2 score of the model.
    """
    
    # Create a linear regression model
    lr = LinearRegression()

    # Fit the model to the training data
    lr.fit(X, y)

    # Predict using the trained model
    Y_pred = lr.predict(X)

    # Evaluate the model
    mse = mean_squared_error(y, Y_pred)
    r2 = r2_score(y, Y_pred)
    
    return lr, mse, r2


def apply_BGLR(features, target, save_eval_df=False, scaling=True, name="BGLR"):
    """
    Applies cross-sectional Linear Regression following the approach by Bartram and Grinblatt (2018) (BGLR) 
    on the provided features and target.

    Args:
        features (pd.DataFrame): Input data containing feature columns.
        target (pd.Series or pd.DataFrame): Target values corresponding to the features.
        save_eval_df (bool, optional): If True, saves the evaluation dataframes to CSV files. Default is False.
        scaling (bool, optional): If True, scales the features using MinMax scaling. Default is True.
        name (str, optional): Name to be used when saving the evaluation dataframes to CSV. Default is "BGLR".
    
    Returns:
        pd.DataFrame: A dataframe with the mean values of Quintile, Mispricing, Next 1M Return, 
                      Next 6M Return, and Next 12M Return, grouped by Quintile.

    Notes:
        - The function splits the data based on unique periods and applies linear regression for each period.
        - Calculates the mispricing signal and adds quintile information.
        - Prints the minimum, maximum, average, and median R^2 scores across all periods.
        - If save_eval_df is True, two evaluation dataframes are saved with names based on the provided 'name'.
    """
    
    # Resetting the index
    df_reset = features.reset_index().drop(columns='tic')
    
    # Get the indices for each period
    groups = df_reset.groupby(df_reset['Date']).indices
    
    # Splitting the features for each period
    features_by_period = {group: features.iloc[indices] for group, indices in groups.items()} 
    target_by_period = {group: target.iloc[indices] for group, indices in groups.items()}
    
    
    lr_list, mse_list, r2_list = [], [], []
    
    for key in features_by_period.keys():
        
        
        if scaling:
            # Apply minmax transform
            features_by_period[key] = MinMaxScaler().fit_transform(features_by_period[key])
        
        target_by_period[key] = pd.DataFrame(target_by_period[key])
        
        
        # Apply linear regression
        lr, mse, r2 = apply_lin_regression(features_by_period[key], target_by_period[key]['Company Market Cap'])
        lr_list.append(lr)
        mse_list.append(mse)
        r2_list.append(r2)
        
        # Add predictions to the target DataFrame
        target_by_period[key]['pred'] = lr.predict(features_by_period[key])
        
        # Add mispricing signal to target DataFrame
        target_by_period[key]['Mispricing'] = (
            target_by_period[key]['pred'] - target_by_period[key]['Company Market Cap']) / target_by_period[key]['Company Market Cap']
        
        # Add quintiles to target DataFrame
        ev.add_quintiles(target_by_period[key], name='Quintile')
        
    print(f"The minimum r2 score for the {len(r2_list)} periods in the dataset is {round(min(r2_list)*100, 2)}% and the maximum is {round(max(r2_list)*100, 2)}%.")
    print(f"The average r2 is {round(np.mean(r2_list)*100, 2)}% and the median is {round(np.median(r2_list)*100, 2)}%.")
    
    
    targets_BG = pd.concat(target_by_period.values())
    targets_BG.reset_index(inplace=True)
    
    next_return_df = pd.read_csv('data/next_returns.csv', index_col=0, parse_dates=[2])
    next_return_df['Date'] = next_return_df['Date'].dt.to_period('M')
    targets_BG = pd.merge(targets_BG, next_return_df, on=['Date', 'tic'], how='left')
    targets_BG_tst = targets_BG[targets_BG['Date'].dt.year >= 2019]

    
    ev2_df = targets_BG_tst[['Quintile', 'Mispricing', 'Next 1M Return', 'Next 6M Return', 'Next 12M Return']].groupby('Quintile').mean()

    
    # save ev2 and targets_BG
    if save_eval_df:
        targets_BG_tst.to_csv(f'model_evaluations_1/{name}.csv')
        ev2_df.to_csv(f'model_evaluations_2/{name}.csv')
    
    
    return ev2_df



