import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def visualise_log_transform(target):
    """
    Visualizes the distribution of market capitalizations before and after log transformation.
    
    This function plots a 1x2 figure, with the left subplot showing the histogram of the 
    original market capitalizations, and the right subplot showing the histogram of the 
    log-transformed market capitalizations.
    
    Args:
        target (pd.Series or np.ndarray): Market capitalization values.
        
    Returns:
        None
    
    Side-effects:
        - Saves the generated figure as a PNG in the "visualisations/" directory.
        - Displays the generated figure.
    
    Notes:
        - A small constant (e.g., 1) is added before log transformation to handle zeros (if any).
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    ax[0].hist(target, bins=100, color='grey')
    
    ax[0].set_xlabel("Market Capitalisations", fontsize=12)
    ax[0].set_ylabel("Frequency", fontsize=12)
    
    ax[0].set_title("Histogram of Market Capitalisations", fontsize=14)
    
    # Add a small constant (e.g., 1) to the target variable to handle zeros (if necessary)
    target_log = target + 1
    
    # Apply the natural logarithm transformation to the target variable
    target_log = np.log(target_log)
    
    ax[1].hist(target_log, bins=100, color='grey')
    
    ax[1].set_xlabel("Market Capitalisations (log-transformed)", fontsize=12)
    ax[1].set_ylabel("Frequency", fontsize=12)

    ax[1].set_title("Histogram of Market Capitalisations (log-transformed)", fontsize=14)
    
    
    plt.savefig("visualisations/Log Transformation Market Capitalisations", dpi=300, bbox_inches='tight')
    
    plt.show()


def visualise_winsorizations(data_win):
    """
    Visualizes the distribution of data after winsorization using violin plots.
    
    This function uses seaborn's violin plot to visualize the distribution of data columns 
    in the provided dataframe after winsorization. A grid of violin plots is created for 
    better visualization.
    
    Args:
        data_win (pd.DataFrame or np.ndarray): Winsorized data with multiple columns.
        
    Returns:
        None
    
    Side-effects:
        - Sets seaborn style/theme.
        - Saves the generated figure as a PNG in the "visualisations/" directory.
        - Displays the generated figure.
    
    Notes:
        - Assumes a maximum of 64 columns in the data.
    """
    
    # Set a theme for seaborn
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=0.9)
    
    
    df = pd.DataFrame(data_win)
    
    # Set the size of the whole figure
    plt.figure(figsize=(20, 18))
    
    n_cols = 8  
    n_rows = 8 
    
    # Create subplots
    for i in range(64):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.violinplot(data=df, x=i, cut=0, palette="deep")#["Grey", "Blue"])
        # plt.legend(loc='lower left')#, title=">$1tn ")
        plt.ylabel("")
        plt.xlabel("")
    
    
    # Adjust the layout for better visualization
    plt.tight_layout()
    plt.savefig("visualisations/Winzorisation_top_bottom_5", dpi=300, bbox_inches='tight')
    
    plt.show()