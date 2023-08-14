import pandas as pd
import numpy as np
from statistics import mode

import matplotlib.pyplot as plt
import seaborn as sns

from hpelm import ELM


import EvalFunctions as ev


def grid_search_elm(Xtrn_minmax, Ytrn, Xval_minmax, Yval, number_neurons=[16, 32, 64, 128, 256, 512, 1024]):
    """
    Conducts a grid search over the number of neurons for an Extreme Learning Machine (ELM) model.
    The ELM model is trained multiple times with different numbers of neurons. The model's performance
    is evaluated based on the spread of returns across quintiles for 1-month, 6-months, and 12-months time horizons.
    The function also plots the performance as a function of the number of neurons for each time horizon.

    Args:
        Xtrn_minmax (numpy.ndarray): The training data, scaled using min-max scaling.
        Ytrn (numpy.ndarray): The actual target values for the training data.
        Xval_minmax (numpy.ndarray): The validation data, scaled using min-max scaling.
        Xval (numpy.ndarray): The actual target values for the validation data.
        number_neurons (list, optional): A list of integers representing the number of neurons to use. 
                                         The default is [16, 32, 64, 128, 256, 512, 1024].

    Returns:
        best_elm (elm.ELM): The best performing ELM model.
        best_neurons (int): The number of neurons in the best performing ELM model.
        best_performance (float): The best average monthly quintile return spread achieved by the ELM model.
    """
    best_elm = None
    best_performance = -float('inf')
    best_neurons = None
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    
    for j in range(10):
        
        q_spread_1M, q_spread_6M, q_spread_12M = [], [], []
    
        for n in number_neurons:
    
            # Define the ELM
            elm = ELM(np.array(Xtrn_minmax).shape[1], 1)  # Adjust to input shape and 1 output
    
            # Add a layer 
            elm.add_neurons(n, 'tanh')
    
            # Train the ELM
            elm.train(np.array(Xtrn_minmax), np.array(Ytrn), 'r')
    
            # Evaluate and save mean quintile results in lists
            mean_summary_df = ev.evaluate_quintile_returns(elm, Xval_minmax, Ytst=Yval, log_transformed=False, save_eval_df=False)
            q_spread_1M.append(mean_summary_df.iloc[4, 1] - mean_summary_df.iloc[0, 1])
            q_spread_6M.append(mean_summary_df.iloc[4, 2] - mean_summary_df.iloc[0, 2])
            q_spread_12M.append(mean_summary_df.iloc[4, 3] - mean_summary_df.iloc[0, 3])
            
            # Compare performance to the best_performance so far
            average_performance = np.mean([q_spread_1M[-1], q_spread_6M[-1]/6, q_spread_12M[-1]/12])
            if average_performance > best_performance:
                best_performance = average_performance
                best_elm = elm
                best_neurons = n
        
    
        ax[0].plot(number_neurons, q_spread_1M)
        ax[1].plot(number_neurons, q_spread_6M)
        ax[2].plot(number_neurons, q_spread_12M)
    
    ax[0].set_title("1 Month Time Horizon", fontsize=12)
    ax[1].set_title("6 Months Time Horizon", fontsize=12)
    ax[2].set_title("12 Months Time Horizon", fontsize=12)
    
    for i in range(3):
        ax[i].set_xlabel("Number of Neurons", fontsize=10)
        ax[i].set_ylabel("Quintile Return Spread", fontsize=10)
    
    plt.suptitle("Grid Search for Number of Neurons (no log, no pca)", fontsize=16)
    
    plt.savefig("visualisations/EML - Grid Search for Number of Neurons", dpi=300, bbox_inches='tight')
    
    print(f"Best performing ELM uses {best_neurons} neurons with an average monthly quintile return spread of {best_performance*100:.2f}%")
    
    plt.show()
    
    
    return best_elm, best_neurons, best_performance






def ELM_activation_grid_search(Xtrn_minmax, Ytrn, Xval_minmax,
                               Yval, activations = ['sigm', 'tanh']):
    """
    Performs a grid search over different activation functions in an Extreme Learning Machine (ELM) model.

    Parameters:
    activations (list of str): The activation functions to be tested. Defaults to [sigm', 'tanh'].

    Returns:
    df (pd.DataFrame): DataFrame containing the spread of returns for each activation function and each time horizon (1 month, 6 months, 12 months).
    """

    q_spread_1M, q_spread_6M, q_spread_12M = [], [], []

    for act in activations:
        # Define the ELM
        elm = ELM(Xtrn_minmax.shape[1], 1)  

        # Add a layer 
        elm.add_neurons(64, act)

        # Train the ELM
        elm.train(Xtrn_minmax, Ytrn, 'r')

        # Evaluate and save mean quintile results in lists
        mean_summary_df = ev.evaluate_quintile_returns(elm, Xval_minmax, Ytst=Yval, log_transformed=False, save_eval_df=False)
        q_spread_1M.append(mean_summary_df.iloc[4, 1] - mean_summary_df.iloc[0, 1])
        q_spread_6M.append(mean_summary_df.iloc[4, 2] - mean_summary_df.iloc[0, 2])
        q_spread_12M.append(mean_summary_df.iloc[4, 3] - mean_summary_df.iloc[0, 3])

    df = {'Activation': activations, 'Spread 1M': q_spread_1M, 'Spread 6M': q_spread_6M, 'Spread 12M': q_spread_12M}
    df = pd.DataFrame(df).set_index('Activation')
    
    return df

