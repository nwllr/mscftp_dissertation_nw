import pandas as pd
import numpy as np
from datetime import datetime

import statsmodels.api as sm
from scipy import stats



def read_FF5FM():
    """
    Reads the Kenneth French research factors from two CSV files and processes it.

    The processing steps include:
    1. Reading the research factors and momentum factors from separate CSV files.
    2. Selecting relevant dates in monthly frequency.
    3. Parsing dates.
    4. Parsing floats.
    5. Merging the momentum factor with the other research factors.
    6. Adding all factors for the 6M and 12M time horizons.

    Returns:
        pandas.DataFrame: A DataFrame containing processed Kenneth French research factors including the momentum factor. 
                          The returned DataFrame also includes the 6-month and 12-month rolling sums of each factor.
    """

    # Read Kenneth French research factors
    FF5F = pd.read_csv('data/FF5F_KennethFrench/F-F_Research_Data_5_Factors_2x3.csv')
    mom = pd.read_csv('data/FF5F_KennethFrench/F-F_Momentum_Factor.csv')
    
    # Extract relevant dates in monthly frequency
    FF5F = FF5F.iloc[438:718]
    mom = mom.iloc[876:-100]
    
    # Parse dates
    for df in [FF5F, mom]:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        df['Date'] = df['Date'].apply(lambda x: pd.Period(year=int(x[:4]), month=int(x[4:]), freq='M'))
    
        # Parse floats
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col])
    
    mom.rename(columns={'Mom   ': 'Mom'}, inplace=True)
    
    # Merge momentum factor with 5 factors
    FF5FM = pd.merge(FF5F, mom, on='Date', how='left')
    
    # Add all factors for the 6M and 12M time horizons
    for factor in FF5FM.columns[1:]:
        FF5FM[f'{factor}_6M'] = FF5FM[factor].rolling(window=6).sum().shift(-5)
        FF5FM[f'{factor}_12M'] = FF5FM[factor].rolling(window=12).sum().shift(-11)
        
    return FF5FM



def read_index_returns():
    """
    Loads the S&P500 returns from a CSV file and processes it.

    The processing steps include:
    1. Parsing dates to monthly frequency.

    Returns:
        pandas.DataFrame: A DataFrame containing S&P500 returns with processed dates.
    """    
    # Load S&P500 returns
    sp500_returns = pd.read_csv('data/next_index_returns.csv', index_col=0, parse_dates=[1])
    sp500_returns['Date'] = sp500_returns['Date'].dt.to_period('M')
    
    return sp500_returns


def calculate_excess_over_RF(df):
    """
    Calculates excess return over risk-free rate for a given DataFrame.

    The function merges the input DataFrame with the Kenneth French 5 factors 
    and computes the excess returns over risk-free rate for the next 1M, 6M, and 12M.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with calculated excess returns.
    """   
    FF5FM = read_FF5FM()

    # Add FF5FM
    df = pd.merge(df, FF5FM, on='Date', how='left')
    df['Next 1M Return (%)'] = df['Next 1M Return']*100
    df['Next 6M Return (%)'] = df['Next 6M Return']*100
    df['Next 12M Return (%)'] = df['Next 12M Return']*100
    df = df.drop(columns=['Next 1M Return', 'Next 6M Return', 'Next 12M Return'])
    # df.dropna(inplace=True)

    df['Next 1M Excess Return (%)'] = df['Next 1M Return (%)'] - df['RF']
    df['Next 6M Excess Return (%)'] = df['Next 6M Return (%)'] - df['RF_6M']
    df['Next 12M Excess Return (%)'] = df['Next 12M Return (%)'] - df['RF_12M']
    
    return df


def calculate_return_over_index(df):
    """
    Returns a quintile grouped DataFrame with an added column containing the average 
    returns over the index.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the DataFrame with added columns and the grouped DataFrame.
    """
    sp500_returns = read_index_returns()
    
    df = pd.merge(df, sp500_returns.drop(columns='Adj Close'), on='Date', how='left')
    df['Next 1M Return over Index (%)'] = df['Next 1M Excess Return (%)'] - df['Next 1M Index Return (%)']
    df['Next 6M Return over Index (%)'] = df['Next 6M Excess Return (%)'] - df['Next 6M Index Return (%)']
    df['Next 12M Return over Index (%)'] = df['Next 12M Excess Return (%)'] - df['Next 12M Index Return (%)']
    
    simple_Q_eval = df[
        ['Quintile', 'Mispricing', 'Next 1M Return (%)', 'Next 1M Excess Return (%)', 'Next 1M Return over Index (%)',
        'Next 6M Return (%)', 'Next 6M Excess Return (%)', 'Next 6M Return over Index (%)',
        'Next 12M Return (%)', 'Next 12M Excess Return (%)', 'Next 12M Return over Index (%)']
    ].groupby('Quintile').mean()
    
    return df, simple_Q_eval


def run_ev3(name):
    """
    Runs the third model evaluation for a given model name.

    The function reads the model results, calculates excess over risk-free rate, 
    computes return over index, and saves the results.

    Args:
        name (str): The name of the model.

    Returns:
        pandas.DataFrame: The DataFrame after the third model evaluation.
    """
    
    # Read model result (ev1)
    try:
        ev3_df =  pd.read_csv(f'model_evaluations_1/{name}.csv', index_col=0)
    except:
        ev3_df =  pd.read_csv(f'model_evaluations_1/{name}', index_col=0)
        
    ev3_df['Date'] = pd.to_datetime(ev3_df['Date'])
    ev3_df['Date'] = ev3_df['Date'].dt.to_period('M')

    ev3_df = calculate_excess_over_RF(ev3_df)
    
    ev3_df, simple_Q_eval = calculate_return_over_index(ev3_df)
    
    simple_Q_eval.to_csv(f'model_evaluations_3/{name}.csv')
    
    return ev3_df

        
def run_portfolio_lvl_regression_CAPM(name, period='1M', save=True):
    """
    Runs and evaluates quintile portfolio level regression for monthly formation dates 
    and 1M, 6M or 12M excess return as dependent variable.

    Independent variable is only the index return and a constant to estimate alpha 
    after controlling for explained returns by the index movement.

    Args:
        name (str): The name of the model.
        period (str, optional): The period for the regression. It can be '1M', '6M', or '12M'. Defaults to '1M'.
        save (bool, optional): Whether to save the regression results. Defaults to True.

    Returns:
        tuple: A tuple containing the list of alphas, p-values, standard errors 
        and degrees of freedom from the regression if save is set to False.
    """
    
    if period not in ['1M', '6M', '12M']:
        raise ValueError("Please select a valid time horizon for the period parameter (1M, 6M or 12M)")
    
    alphas = []
    p_values = []
    std_errs = []
    defree = []
    
    df = run_ev3(name)
    
    if period == '1M':
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df['RF']
    else:
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df[f'RF_{period}']
    
    
    # Group by date and quintile, calculate average excess return
    df_grouped = df.groupby(['Date', 'Quintile'])[f'Next {period} Excess Return (%)'].mean().reset_index()

    # Merge back with the original dataframe to get factors
    df_merged = pd.merge(df_grouped, df[['Date', f'Next {period} Index Return (%) - RF']], on='Date', how='left')

    # Drop duplicates
    df_merged = df_merged.drop_duplicates()
    
    # Select relevant column for selected period and drop NAs
    df_merged = df_merged[['Date', 'Quintile', f'Next {period} Index Return (%) - RF', f'Next {period} Excess Return (%)']].dropna()

    # Run regression
    for quintile in range(1, 6):
        df_quintile = df_merged[df_merged['Quintile'] == quintile]
        X = df_quintile[f'Next {period} Index Return (%) - RF']
        X = sm.add_constant(X)
        y = df_quintile[f'Next {period} Excess Return (%)']
        model = sm.OLS(y, X).fit()
        
        alphas.append(model.params[0])
        p_values.append(model.pvalues[0])
        std_errs.append(model.bse[0])
        defree.append(model.df_resid)
        
        
        if save:
            print('\n\n')
            print(f'Regression results for quintile {int(quintile)}:')
            print('\n')
            print(model.summary())
            
            with open(f'alpha_CAPM_QPortfolio_lvl/{name}', 'a') as file:
                if int(quintile) == 1:
                    file.write('\n' + 80*'#')
                    file.write(f"\n\nTIMESTAMP: {datetime.now()}\n")
                    
                file.write(f'\n\nRegression results for quintile {int(quintile)} ({period}):\n')
                file.write(str(model.summary()))
        
    if not save:
        return alphas, p_values, std_errs, defree
           
    
def run_portfolio_lvl_regression_FF5FM(name, period='1M', save=True):
    """
    Runs and evaluates quintile portfolio level regression for monthly formation dates and 1M, 6M or 12M excess return as dependent variable.

    The independent variables are the Fama-French five factors and a constant to estimate alpha after controlling for returns explained by these factors.

    Args:
        name (str): The name of the model.
        period (str, optional): The period for the regression. It can be '1M', '6M', or '12M'. Defaults to '1M'.
        save (bool, optional): Whether to save the regression results. Defaults to True.

    Raises:
        ValueError: If the provided period is not '1M', '6M', or '12M'.

    Returns:
        tuple: A tuple containing the list of alphas, p-values, standard errors 
        and degrees of freedom from the regression if save is set to False.
    """

    if period not in ['1M', '6M', '12M']:
        raise ValueError("Please select a valid time horizon for the period parameter (1M, 6M or 12M)")
    
    alphas = []
    p_values = []
    std_errs = []
    defree = []
    
    df = run_ev3(name)
    
    
    if period == '1M':    
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df['RF']
        
        # Group by date and quintile, calculate average excess return
        df_grouped = df.groupby(['Date', 'Quintile'])[
            f'Next {period} Excess Return (%)'].mean().reset_index()
        
        # Merge back with the original dataframe to get factors
        df_merged = pd.merge(
            df_grouped, df[['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']],
            on='Date', how='left')
    
    else:
        df[f'Next {period} Index Return (%) - RF'] = df[
            f'Next {period} Index Return (%)'] - df[f'RF_{period}']
        
        # Group by date and quintile, calculate average excess return
        df_grouped = df.groupby(['Date', 'Quintile'])[
            f'Next {period} Excess Return (%)'].mean().reset_index()
        
        # Merge back with the original dataframe to get factors
        df_merged = pd.merge(
            df_grouped, df[['Date', f'Mkt-RF_{period}', f'SMB_{period}',
                             f'HML_{period}', f'RMW_{period}', f'CMA_{period}',
                             f'Mom_{period}']], on='Date', how='left')

    # Drop duplicates
    df_merged = df_merged.drop_duplicates()
    
    # Select relevant column for selected period and drop NAs
    if period == '1M':
        df_merged = df_merged[['Date', 'Quintile', 'Mkt-RF', 'SMB', 'HML', 'RMW',
                               'CMA', 'Mom', f'Next {period} Excess Return (%)']].dropna()
    else:
        df_merged = df_merged[['Date', 'Quintile', f'Mkt-RF_{period}', f'SMB_{period}',
                         f'HML_{period}', f'RMW_{period}', f'CMA_{period}',
                         f'Mom_{period}', f'Next {period} Excess Return (%)']].dropna()

    # Run regression
    for quintile in range(1, 6):
        df_quintile = df_merged[df_merged['Quintile'] == quintile]
        
        if period == '1M':
            X = df_quintile[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']]
        else:
            X = df_quintile[[f'Mkt-RF_{period}', f'SMB_{period}',
                             f'HML_{period}', f'RMW_{period}', 
                             f'CMA_{period}', f'Mom_{period}']]
        
        X = sm.add_constant(X)
        y = df_quintile[f'Next {period} Excess Return (%)']
        
        model = sm.OLS(y, X).fit()
        
        alphas.append(model.params[0])
        p_values.append(model.pvalues[0])
        std_errs.append(model.bse[0])
        defree.append(model.df_resid)
        
        
        if save:
            print('\n\n')
            print(f'Regression results for quintile {int(quintile)}:')
            print('\n')
            print(model.summary())
            
            with open(f'alpha_FF5FM_QPortfolio_lvl/{name}', 'a') as file:
                if int(quintile) == 1:
                    file.write('\n' + 80*'#')
                    file.write(f"\n\nTIMESTAMP: {datetime.now()}\n")
                    
                file.write(f'\n\nRegression results for quintile {int(quintile)} ({period}):\n')
                file.write(str(model.summary()))
        
    if not save:
        return alphas, p_values, std_errs, defree



def run_stock_lvl_regression_CAPM(name, period='1M', save=True):
    """
    Runs and evaluates regression at the stock level for monthly formation dates and 1M, 6M or 12M excess return as dependent variable.

    The independent variable is the index return and a constant to estimate alpha after controlling for returns explained by the index movement.

    Args:
        name (str): The name of the model.
        period (str, optional): The period for the regression. It can be '1M', '6M', or '12M'. Defaults to '1M'.
        save (bool, optional): Whether to save the regression results. Defaults to True.

    Raises:
        ValueError: If the provided period is not '1M', '6M', or '12M'.

    Returns:
        tuple: A tuple containing the list of alphas, p-values, standard errors 
        and degrees of freedom from the regression if save is set to False.
    """

    if period not in ['1M', '6M', '12M']:
        raise ValueError("Please select a valid time horizon for the period parameter (1M, 6M or 12M)")
    
    alphas = []
    p_values = []
    std_errs = []
    defree = []
    
    df = run_ev3(name)
    
    if period == '1M':
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df['RF']
    else:
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df[f'RF_{period}']
    
    
    
    # Select relevant column for selected period and drop NAs
    df = df[['Date', 'Quintile', f'Next {period} Index Return (%) - RF', f'Next {period} Excess Return (%)']].dropna()

    # Run regression
    for quintile in range(1, 6):
        
        X = df[df['Quintile'] == quintile][f'Next {period} Index Return (%) - RF']
        X = sm.add_constant(X)
        y = df[df['Quintile'] == quintile][f'Next {period} Excess Return (%)']
        model = sm.OLS(y, X).fit()
        
        alphas.append(model.params[0])
        p_values.append(model.pvalues[0])
        std_errs.append(model.bse[0])
        defree.append(model.df_resid)
        
        
        if save:
            print('\n\n')
            print(f'Regression results for quintile {int(quintile)}:')
            print('\n')
            print(model.summary())
            
            with open(f'alpha_CAPM_stock_lvl/{name}', 'a') as file:
                if int(quintile) == 1:
                    file.write('\n' + 80*'#')
                    file.write(f"\n\nTIMESTAMP: {datetime.now()}\n")
                    
                file.write(f'\n\nRegression results for quintile {int(quintile)} ({period}):\n')
                file.write(str(model.summary()))
        
    if not save:
        return alphas, p_values, std_errs, defree


def run_stock_lvl_regression_FF5FM(name, period='1M', save=True):
    """
    Runs and evaluates regression at the stock level for monthly formation dates and 1M, 6M or 12M excess return as dependent variable.

    The independent variables are the Fama-French five factors and a constant to estimate alpha after controlling for returns explained by these factors.

    Args:
        name (str): The name of the model.
        period (str, optional): The period for the regression. It can be '1M', '6M', or '12M'. Defaults to '1M'.
        save (bool, optional): Whether to save the regression results. Defaults to True.

    Raises:
        ValueError: If the provided period is not '1M', '6M', or '12M'.

    Returns:
        tuple: A tuple containing the list of alphas, p-values, standard errors 
        and degrees of freedom from the regression if save is set to False.
    """

    if period not in ['1M', '6M', '12M']:
        raise ValueError("Please select a valid time horizon for the period parameter (1M, 6M or 12M)")
    
    alphas = []
    p_values = []
    std_errs = []
    defree = []
    
    df = run_ev3(name)
    
    if period == '1M':
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df['RF']
    else:
        df[f'Next {period} Index Return (%) - RF'] = df[f'Next {period} Index Return (%)'] - df[f'RF_{period}']
    
    
    # Select relevant column for selected period and drop NAs
    if period == '1M':
        df = df[['Date', 'Quintile', 'Mkt-RF', 'SMB', 'HML', 'RMW',
                               'CMA', 'Mom', f'Next {period} Excess Return (%)']].dropna()
    else:
        df = df[['Date', 'Quintile', f'Mkt-RF_{period}', f'SMB_{period}',
                         f'HML_{period}', f'RMW_{period}', f'CMA_{period}',
                         f'Mom_{period}', f'Next {period} Excess Return (%)']].dropna()
        
        
    # Run regression
    for quintile in range(1, 6):
        
        if period == '1M':
            X = df[df['Quintile'] == quintile][
                ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']]
        else:
            X = df[df['Quintile'] == quintile][
                [f'Mkt-RF_{period}', f'SMB_{period}',
                 f'HML_{period}', f'RMW_{period}', f'CMA_{period}',
                 f'Mom_{period}']]
            
        X = sm.add_constant(X)
        y = df[df['Quintile'] == quintile][f'Next {period} Excess Return (%)']
        model = sm.OLS(y, X).fit()
        
        alphas.append(model.params[0])
        p_values.append(model.pvalues[0])
        std_errs.append(model.bse[0])
        defree.append(model.df_resid)
        
        
        if save:
            print('\n\n')
            print(f'Regression results for quintile {int(quintile)}:')
            print('\n')
            print(model.summary())
            
            with open(f'alpha_FF5FM_stock_lvl/{name}', 'a') as file:
                if int(quintile) == 1:
                    file.write('\n' + 80*'#')
                    file.write(f"\n\nTIMESTAMP: {datetime.now()}\n")
                    
                file.write(f'\n\nRegression results for quintile {int(quintile)} ({period}):\n')
                file.write(str(model.summary()))
        
    if not save:
        return alphas, p_values, std_errs, defree
    



def create_xlsx_alpha_analysis(alpha_summary, factors, lvl, save_as=None):
    """
    This function takes in a dataframe containing alpha values and their corresponding p-values. It modifies the dataframe to
    annotate the alpha values based on the significance level of their p-values. It uses the following convention for
    annotations:
    - '***': p-value < 0.01
    - '**': 0.01 <= p-value < 0.05
    - '*': 0.05 <= p-value < 0.1
    - No annotation: p-value >= 0.1

    After annotation, the function optionally saves the dataframe to an Excel file in the 'alpha_capm' directory.

    Args:
        alpha_summary (pandas.DataFrame): A dataframe where the first half of the columns contain alpha values and the second half contain corresponding p-values.
        save_as (str, optional): If provided, the modified dataframe is saved as an Excel file with this name. The filename is prefixed with 'AN2_'. The file is saved in the 'alpha_capm' directory.

    Returns:
        None
    """
    
    alp = round(alpha_summary.iloc[:,:int(len(alpha_summary.columns)/2)], 2)
    
    alpha_summary_2 = alpha_summary.copy()
    
    for row in range(len(alp.index)):
        for col in range(len(alp.columns)):
            
            if (alpha_summary.iloc[row, len(alp.columns)+col] < 0.01) == True:
                alpha_summary_2.iloc[row, col] = str(alp.iloc[row, col]) + '***'
                
            elif (alpha_summary.iloc[row, len(alp.columns)+col] < 0.05) == True:
                alpha_summary_2.iloc[row, col] = str(alp.iloc[row, col]) + '**'
                
            elif (alpha_summary.iloc[row, len(alp.columns)+col] < 0.1) == True:
                alpha_summary_2.iloc[row, col] = str(alp.iloc[row, col]) + '*'
                
            else:
                alpha_summary_2.iloc[row, col] = str(alp.iloc[row, col])
    
    if save_as is not None:
        alpha_summary_2.to_excel(f'alpha_{factors}_{lvl}_lvl/AN2_{save_as}.xlsx')
        
        
    
def est_alphas_summary(model_names, factors, lvl, save_as=None):
    """
    Computes the alphas for either portfolio or stock level abnormal returns, unexplained by a given factors model (either CAPM or FF5FM).

    The function computes alphas for different periods (1M, 6M, 12M) and for different quintiles. It also computes the p-values associated with the computed alphas.

    Args:
        model_names (list): A list of model names for which to compute the alphas.
        factors (str): The name of the factors model to use for the alpha computation. It must be either 'CAPM' or 'FF5FM'.
        lvl (str): The level at which to compute the alphas. It must be either 'QPortfolio' for portfolio level or 'stock' for stock level.
        save_as (str, optional): If provided, the alpha summary is saved as an Excel file with this name.

    Raises:
        ValueError: If the provided factors model name is not 'CAPM' or 'FF5FM', or the level is not 'QPortfolio' or 'stock'.

    Returns:
        pandas.DataFrame: A DataFrame containing the alpha summary. The DataFrame includes alpha values, p-values, and max p-values for different periods and quintiles.

    """


    if factors not in ['CAPM', 'FF5FM']:
        raise ValueError("Please select a valid value for factors (CAPM or FF5FM).")
    elif lvl not in ['QPortfolio', 'stock']:
        raise ValueError("Please select a valid value for level (QPortfolio or stock).")
    
    periods = ['1M', '6M', '12M']
    
    alphas_dict = {"Q1 1M": [], "Q2 1M": [], "Q3 1M": [], 
                   "Q4 1M": [], "Q5 1M": [], "Q5-Q1 1M": [],
                   "Q1 6M": [], "Q2 6M": [], "Q3 6M": [], 
                   "Q4 6M": [], "Q5 6M": [], "Q5-Q1 6M": [],
                   "Q1 12M": [], "Q2 12M": [], "Q3 12M": [], 
                   "Q4 12M": [], "Q5 12M": [], "Q5-Q1 12M": [],
                   "Q1 1M p-value": [], "Q2 1M p-value": [], "Q3 1M p-value": [], 
                   "Q4 1M p-value": [], "Q5 1M p-value": [], "Q5-Q1 1M p-value": [],
                   "Q1 6M p-value": [], "Q2 6M p-value": [], "Q3 6M p-value": [], 
                   "Q4 6M p-value": [], "Q5 6M p-value": [], "Q5-Q1 6M p-value": [],
                   "Q1 12M p-value": [], "Q2 12M p-value": [], "Q3 12M p-value": [], 
                   "Q4 12M p-value": [], "Q5 12M p-value": [], "Q5-Q1 12M p-value": []}
    
    for model in model_names:
        for p in periods:
            
            if lvl == 'QPortfolio':
                if factors == 'CAPM':
                    alphas, p_values, std_errs, defree = run_portfolio_lvl_regression_CAPM(model, period=p, save=False)
                elif factors == 'FF5FM':
                    alphas, p_values, std_errs, defree = run_portfolio_lvl_regression_FF5FM(model, period=p, save=False)
            
            elif lvl == 'stock':
                if factors == 'CAPM':
                    alphas, p_values, std_errs, defree = run_stock_lvl_regression_CAPM(model, period=p, save=False)
                elif factors == 'FF5FM':
                    alphas, p_values, std_errs, defree = run_stock_lvl_regression_FF5FM(model, period=p, save=False)
            
            
            for i in range(1, 6):
                alphas_dict[f"Q{i} {p}"].append(alphas[i-1])
                alphas_dict[f"Q{i} {p} p-value"].append(p_values[i-1])
                
            alphas_dict[f"Q5-Q1 {p}"].append(alphas[4] - alphas[0])
            
            # alphas_dict[f"Q5-Q1 {p} max p"].append(max(p_values[4], p_values[0]))
            
            # Calculate t-statistic for Q5-Q1
            alpha_diff = alphas[4] - alphas[0]
            std_err_diff = np.sqrt(std_errs[4]**2 + std_errs[0]**2)  # assuming Q5 and Q1 are independent
            t_stat_diff = alpha_diff / std_err_diff if std_err_diff != 0 else np.nan
            
            # Calculate p-value (one-sided as only interested in larger than 0)
            p_value_diff = (1 - stats.t.cdf(abs(t_stat_diff), min(defree[0], defree[4]))) # conservative estimation (minimum degrees of freedom)
            alphas_dict[f"Q5-Q1 {p} p-value"].append(p_value_diff)
    
    alpha_summary = pd.DataFrame(alphas_dict, index=model_names)
    
    create_xlsx_alpha_analysis(alpha_summary, factors=factors, lvl=lvl, save_as=save_as)
    
    if save_as is not None:
        alpha_summary.to_excel(f'alpha_{factors}_{lvl}_lvl/AN1_{save_as}.xlsx')
    
    return alpha_summary



def calculate_percentage_change_xlsx(full_data_models, reduced_data_models, save_as=None):
    """
    Calculate percentage change in alpha values between models trained on full and reduced datasets.
    
    For each model, this function computes the percentage change in estimated alphas
    between the version trained on full data and its counterpart trained on reduced data.
    The results are optionally saved in an Excel file.

    Args:
        full_data_models (list of str): List of model names trained on the full dataset.
        reduced_data_models (list of str): Corresponding list of model names trained on the reduced dataset.
        save_as (str, optional): Prefix for saving the results in Excel format. If None, results are not saved.
                                 Output filename format is 'alpha_less_data/AN3_{factor}_{save_as}.xlsx'.

    Returns:
        None

    Raises:
        ValueError: If the lengths of `full_data_models` and `reduced_data_models` are not the same.

    Notes:
        - This function internally calls `est_alphas_summary` to compute summary statistics 
          for alpha values of the models.
        - The output dataframe contains computed percentage change values, and its values 
          are rounded to two decimal places.
        - The function works for factors 'CAPM' and 'FF5FM'.
    """

    if len(full_data_models) != len(reduced_data_models):
        raise ValueError("Please provide two lists with the same lengths")
    


    for f in ['CAPM', 'FF5FM']:
        df = est_alphas_summary(full_data_models + reduced_data_models, factors=f, lvl='stock', save_as=None)
        df = df.T
        
        for i in range(len(full_data_models)):
            df[f"{reduced_data_models[i][:-12]} - Percentage Change"] = \
                (df[reduced_data_models[i]] - df[full_data_models[i]]) / abs(df[full_data_models[i]]) * 100

        df = df.T
        
        
        for row in range(len(df.index)):
            for col in range(len(df.columns)):
                    
                df.iloc[row, col] = str(round(df.iloc[row, col], 2))
        
        if save_as is not None:
            df.to_excel(f'alpha_less_data/AN3_{f}_{save_as}.xlsx')
        
    


















