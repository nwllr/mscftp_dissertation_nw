import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from keras.wrappers.scikit_learn import KerasRegressor


import EvalFunctions as ev


# function to create model, required for KerasRegressor
def create_MLP(input_dim, hidden_layers=1, neurons=32):
    """
    Creates a multilayer perceptron (MLP) model using Keras Sequential API.
    
    Args:
        input_dim (int): The number of input features.
        hidden_layers (int, optional): The number of hidden layers. Default is 1.
        neurons (int, optional): The number of neurons in each hidden layer. Default is 32.
    
    Returns:
        model (keras.models.Sequential): The created MLP model.
    """

    model = Sequential()
    model.add(Dense(neurons, input_dim=input_dim, activation='relu'))
    
    for i in range(hidden_layers-1): 
        model.add(Dense(neurons, activation='relu'))
        
    model.add(Dense(1))
    adam = Adam(clipvalue=1)  # gradient clipping to prevent exploding gradients
    model.compile(loss='mean_squared_error', optimizer=adam)
    
    return model


def topological_grid_search_CV_MLP(Xtrn, Ytrn, hidden_layers = [1, 2, 3],
                                neurons = [16, 32, 64, 128, 512]):
    """
    Conducts a grid search over the number of hidden layers and neurons for a MLP model using cross validation.

    Args:
        Xtrn (numpy.ndarray): The training data.
        Ytrn (numpy.ndarray): The target values for the training data.
        hidden_layers (list, optional): A list of integers representing the number of hidden layers to consider. Default is [1, 2, 3].
        neurons (list, optional): A list of integers representing the number of neurons to consider. Default is [16, 32, 64, 128, 512].

    Returns:
        best_mlp (KerasRegressor): The best performing MLP model.
        best_mae (float): The minimum mean absolute error achieved.
        best_neurons (int): The number of neurons in the best performing MLP model.
        best_n_layers (int): The number of hidden layers in the best performing MLP model.
    """

    best_mlp = None
    best_mae = float('inf')
    best_neurons = None
    best_n_layers = None
    counter = 0
    maes = []
    
    for n_layers in hidden_layers:
        for n_neurons in neurons:
            counter += 1
            print(f"\n\nCombination {counter} of {len(hidden_layers)*len(neurons)}: Fitting {n_layers} hidden layers with {n_neurons} neurons:\n")
            
            mlp = KerasRegressor(
                build_fn=create_MLP, input_dim=Xtrn.shape[1],
                hidden_layers=n_layers, neurons=n_neurons, epochs=10,
                batch_size=16, verbose=1)
            
            # Compute cross-validated MAE
            mae = cross_val_score(mlp, Xtrn, Ytrn,
                                  cv=3, scoring=make_scorer(mean_absolute_error)).mean()
            
            print(mae)
            maes.append(mae)
    
            if mae < best_mae:
                best_mae = mae
                best_mlp = mlp
                best_neurons = n_neurons
                best_n_layers = n_layers


    # Report Results
    print(f"Best performing MLP has {best_n_layers} hidden layers with {best_neurons} neurons and achieved an mean absolute error of {best_mae:.2f}")
    
    return best_mlp, best_mae, best_neurons, best_n_layers



def topo_grid_MLP_2(Xtrn, Ytrn, Xval, Yval, hidden_layers = [1, 2, 3],
                    neurons = [16, 32, 64, 128, 512]):
    """
    Conducts a grid search over the number of hidden layers and neurons for a MLP model using a validation set.

    Args:
        Xtrn (numpy.ndarray): The training data.
        Ytrn (numpy.ndarray): The target values for the training data.
        Xval (numpy.ndarray): The validation data.
        Yval (numpy.ndarray): The target values for the validation data.
        hidden_layers (list, optional): A list of integers representing the number of hidden layers to consider. Default is [1, 2, 3].
        neurons (list, optional): A list of integers representing the number of neurons to consider. Default is [16, 32, 64, 128, 512].

    Returns:
        best_mlp (keras.models.Model): The best performing MLP model.
        best_performance (float): The best average monthly quintile return spread achieved.
        best_neurons (int): The number of neurons in the best performing MLP model.
        best_n_layers (int): The number of hidden layers in the best performing MLP model.
    """

    
    best_mlp = None
    best_performance = -float('inf')
    best_neurons = None
    best_n_layers = None
    counter = 0
    q_spread_1M, q_spread_6M, q_spread_12M = [], [], []
    
    
    for n_layers in hidden_layers:
        for n_neurons in neurons:
            
            counter += 1
            print(f"\n\nCombination {counter} of {len(hidden_layers)*len(neurons)}: Fitting {n_layers} hidden layers with {n_neurons} neurons:\n")
            
            mlp = create_MLP(Xtrn.shape[1], hidden_layers=n_layers, neurons=n_neurons)
            mlp.fit(Xtrn, Ytrn, epochs=20, batch_size=16, verbose=1)
            
            # Evaluate and save mean quintile results in lists
            mean_summary_df = ev.evaluate_quintile_returns(mlp, Xval, Ytst=Yval, log_transformed=False, save_eval_df=False)
            q_spread_1M.append(mean_summary_df.iloc[4, 1] - mean_summary_df.iloc[0, 1])
            q_spread_6M.append(mean_summary_df.iloc[4, 2] - mean_summary_df.iloc[0, 2])
            q_spread_12M.append(mean_summary_df.iloc[4, 3] - mean_summary_df.iloc[0, 3])
            
            # Compare performance to the best_performance so far
            average_performance = np.mean([q_spread_1M[-1], q_spread_6M[-1]/6, q_spread_12M[-1]/12])
            if average_performance > best_performance:
                best_performance = average_performance
                best_mlp = mlp
                best_neurons = n_neurons
                best_n_layers = n_layers
        
        
    # Report Results
    print(f"Best performing MLP has {best_n_layers} hidden layers with {best_neurons} neurons and achieves an average monthly quintile return spread of {best_performance*100:.2f}%")
    
    return best_mlp, best_performance, best_neurons, best_n_layers


def apply_MLP_3HL(Xtrn, Ytrn, Xval, Yval, Xtst, Ytst, neurons=512, 
                  log_transformed=False, save_eval_df=False, name="NN 3HL512N"):
    """
    Trains a multilayer perceptron (MLP) model with three hidden layers.

    Args:
        Xtrn (numpy.ndarray): The training data.
        Ytrn (numpy.ndarray): The target values for the training data.
        Xval (numpy.ndarray): The validation data.
        Yval (numpy.ndarray): The target values for the validation data.
        Xtst (numpy.ndarray): The test data.
        Ytst (numpy.ndarray): The target values for the test data.
        neurons (int, optional): The number of neurons in each hidden layer. Default is 512.
        log_transformed (bool, optional): Whether the target values are log-transformed. Default is False.
        save_eval_df (bool, optional): Whether to save the evaluation dataframe. Default is False.
        name (str, optional): The name of the model. Default is 'NN 3HL512N'.

    Returns:
        ev2_df (pandas.DataFrame): A dataframe that contains the evaluation results of the model.
    """

    # Define the model using the Functional API
    input_layer = Input(shape=Xtrn.shape[1]) 
    hidden_layer_1 = Dense(neurons, activation='relu')(input_layer)
    hidden_layer_2 = Dense(neurons, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(neurons, activation='relu')(hidden_layer_2)
    output_layer = Dense(1)(hidden_layer_3) 
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    adam = Adam(clipvalue=1)  # gradient clipping to avoid exploiding gradient problem
    model.compile(loss='mean_squared_error', optimizer=adam)
        
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Train the model
    model.fit(Xtrn, Ytrn, 
                   epochs=500,  
                   batch_size=64, 
                   validation_data=(Xval, Yval),
                   callbacks=[early_stopping],
                   verbose=1)
    
    
    ev2_df = ev.evaluate_quintile_returns(model,
                                          Xtst,
                                          Ytst,
                                          log_transformed=log_transformed,
                                          save_eval_df=save_eval_df,
                                          name=name)
    
    if save_eval_df:
        ev2_df.to_csv(f'model_evaluations_2/{name}.csv')
        
        
    return ev2_df




