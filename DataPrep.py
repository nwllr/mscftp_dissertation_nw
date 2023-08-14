import pandas as pd
import numpy as np
import os
import yfinance as yf

def read_all_csvs_in_folder(folder_path):
    """
    Reads all the CSV files in a given directory and returns a list of dataframes.
    
    Args:
        folder_path (str): Path to the folder containing the CSV files.
    
    Returns:
        list[pd.DataFrame]: A list of dataframes, each dataframe corresponding to one CSV file.
    
    Note:
        It is assumed that all the files in the given directory with a '.csv' extension are valid CSV files.
    """
    # Get a list of all the csv files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Initialize an empty dataframe
    df_list = []

    # Loop through all the files and read them into a dataframe
    for file in files:
        file_path = os.path.join(folder_path, file)
        df_list.append(pd.read_csv(file_path))
        
        # df = pd.concat([df, temp_df], ignore_index=True)

    return df_list


# Define a function to apply the resampling and filling to each group
def resample_group(group):
    """
    Resamples a dataframe's date to monthly frequency and forward fills missing values.

    Args:
        group (pd.DataFrame): The dataframe to be resampled. It should have a 'Date' column.

    Returns:
        pd.DataFrame: A dataframe with monthly resampled data.

    Note:
        The function sets the 'Date' column as the index before resampling.
    """
    group.set_index('Date', inplace=True)
    group_resampled = group.resample('M')
    group_resampled = group_resampled.ffill()
    group_resampled.reset_index(inplace=True)
    return group_resampled


# Define a function to apply the shift operation to each group
def shift_group(group):
    """
    Shifts specified columns in a dataframe by 2 rows.

    Args:
        group (pd.DataFrame): The dataframe with columns to be shifted.

    Returns:
        pd.DataFrame: A dataframe with specified columns shifted.

    Note:
        The global variable 'cols_to_shift' should be defined and contain a list of column names to shift.
    """
    group[cols_to_shift] = group[cols_to_shift].shift(2)
    return group


def calculate_next_1M_return(next_return_df):
    """
    Calculates the next month's return based on the company's market capitalization.

    Args:
        next_return_df (pd.DataFrame): Dataframe containing the 'Company Market Cap' column.

    Returns:
        pd.DataFrame: A dataframe with a new 'Next 1M Return' column added.

    Note:
        This function calculates the next month's return as (Next Market Cap / Current Market Cap) - 1.
    """
    
    next_return_df['Next Market Cap'] = next_return_df['Company Market Cap'].shift(-1)
    next_return_df['Next 1M Return'] = (next_return_df['Next Market Cap']/next_return_df['Company Market Cap']) - 1
    next_return_df.drop(columns='Next Market Cap', inplace=True)
    
    return next_return_df


def calculate_next_6M_return(next_return_df):
    """
    Calculates the next six-months return based on the company's market capitalization.

    Args:
        next_return_df (pd.DataFrame): Dataframe containing the 'Company Market Cap' column.

    Returns:
        pd.DataFrame: A dataframe with a new 'Next 1M Return' column added.

    Note:
        This function calculates the next month's return as (Next Market Cap / Current Market Cap) - 1.
    """
    
    next_return_df['Next Market Cap'] = next_return_df['Company Market Cap'].shift(-6)
    next_return_df['Next 6M Return'] = (next_return_df['Next Market Cap']/next_return_df['Company Market Cap']) - 1
    next_return_df.drop(columns='Next Market Cap', inplace=True)
    
    return next_return_df


def calculate_next_12M_return(next_return_df):
    """
    Calculates the next year's return based on the company's market capitalization.

    Args:
        next_return_df (pd.DataFrame): Dataframe containing the 'Company Market Cap' column.

    Returns:
        pd.DataFrame: A dataframe with a new 'Next 1M Return' column added.

    Note:
        This function calculates the next month's return as (Next Market Cap / Current Market Cap) - 1.
    """
    
    next_return_df['Next Market Cap'] = next_return_df['Company Market Cap'].shift(-12)
    next_return_df['Next 12M Return'] = (next_return_df['Next Market Cap']/next_return_df['Company Market Cap']) - 1
    next_return_df.drop(columns='Next Market Cap', inplace=True)
    
    return next_return_df



# ## Compustat Data (Income Statement and Balance Sheet Items from BG)
acc_items_raw = pd.read_csv('data/compustat/BG_27_accounting_items.csv', parse_dates=[1])
acc_items = acc_items_raw.drop(columns=acc_items_raw.columns[-26:])

for col in acc_items.columns:
    if acc_items[col].nunique() == 1:
        acc_items.drop(columns=col, inplace=True)
        

sp500_tics = pd.read_csv('data/refinitiv/sp500_tickers.csv')
sp500_rics = pd.read_csv('data/refinitiv/sp500_rics.csv')
sp500_identifiers = sp500_rics[['Constituent Name', 'Constituent RIC']]
sp500_identifiers['tic'] = sp500_tics['0']


acc_items = pd.merge(acc_items, sp500_identifiers, on='tic', how='left')
acc_items.rename(columns={'datadate': 'Date', 'Constituent RIC': 'ric'}, inplace=True)

# ## Add Cashflow Items

folder_path = 'data/refinitiv/cashflow'
cashflow_items = read_all_csvs_in_folder(folder_path)


# Dropping duplicate rows for items where only yearly data is available
for i in range(len(cashflow_items)):
    cashflow_items[i].drop_duplicates(inplace=True)

# Initialise merged_income_df
merged_cashflow_df = cashflow_items[0].drop(columns='RIC')

# Merge the remaining balance sheet items onto the initialised merged_income_df
for i in range(1,len(cashflow_items)):
    # Merge the dataframes on the 'Date' column
    merged_cashflow_df = merged_cashflow_df.merge(
        cashflow_items[i].drop(columns='RIC'),
        on=['Date', 'Instrument'], how='outer')


merged_cashflow_df.rename(columns={'Instrument': 'ric'}, inplace=True)
merged_cashflow_df['Date'] = pd.to_datetime(merged_cashflow_df['Date'])

# Convert 'Date' column to timezone-unaware datetime
merged_cashflow_df['Date'] = merged_cashflow_df['Date'].dt.tz_convert(None)

acc_items_full = pd.merge(acc_items, merged_cashflow_df, on=['ric', 'Date'], how='left')




forward_filled = (acc_items_full[['ric',
                                  'Net Cash Flow from Financing Activities', 
                                  'Net Cash Flow from Investing Activities',
                                  'Net Cash Flow from Operating Activities']]
                  .groupby('ric').apply(lambda group: group.ffill()))

# Reset the index
forward_filled.reset_index(level=0, drop=True, inplace=True)
forward_filled.drop(columns='ric', inplace=True)

acc_items_full[['Net Cash Flow from Financing Activities', 
                'Net Cash Flow from Investing Activities',
                'Net Cash Flow from Operating Activities']] = forward_filled



acc_items_full[acc_items_full['tic'] == 'AAPL']


acc_items_clean = acc_items_full.copy()
acc_items_clean['Date'] = acc_items_clean['Date'].dt.to_period('M')
acc_items_clean.drop(columns=['gvkey', 'fyearq', 'fqtr', 'fyr', 'datacqtr', 'datafqtr'], inplace=True)


# ## Macroeconomic Data

folder_path = 'data/fred_oecd/Macroeconomics'

macro_raw = read_all_csvs_in_folder(folder_path)


for i in range(len(macro_raw)):
    
    # This is for the six CSVs sourced from FRED
    if 'DATE' in macro_raw[i].columns:
        macro_raw[i]['DATE'] = pd.to_datetime(macro_raw[i]['DATE'])
        macro_raw[i].rename(columns={'DATE': 'Date'}, inplace=True)
        
    
    # This is for the two CSVs sourced from OECD
    else: 
        macro_raw[i]['TIME'] = pd.to_datetime(macro_raw[i]['TIME'])
        macro_raw[i] = macro_raw[i][
            macro_raw[i]['LOCATION'] == 'USA'][
            ['TIME', 'Value']].rename(columns={'TIME': 'Date'})
        
    macro_raw[i]['Date'] = macro_raw[i]['Date'].dt.to_period('M')
    


# Merge the dataframes on the 'Date' column
macros = macro_raw[0]

for i in range(1, len(macro_raw)):
    macros = pd.merge(macros, macro_raw[i], on='Date', how='outer')

macros = macros[macros['Date'].dt.year > 1999]


macros.sort_values(by='Date', inplace=True, na_position='first')
macros.ffill(inplace=True)



# ## Energy and Commodities

folder_path = 'data/fred_oecd/Energy'

energy_raw = read_all_csvs_in_folder(folder_path)


for i in range(len(energy_raw)):
    
    energy_raw[i]['DATE'] = pd.to_datetime(energy_raw[i]['DATE'])
    energy_raw[i].rename(columns={'DATE': 'Date'}, inplace=True)  
    energy_raw[i]['Date'] = energy_raw[i]['Date'].dt.to_period('M')


# Merge the dataframes on the 'Date' column
energy = energy_raw[0]

for i in range(1, len(energy_raw)):
    energy = pd.merge(energy, energy_raw[i], on='Date', how='outer')

energy.sort_values(by='Date', inplace=True)
energy.ffill(inplace=True)



folder_path = 'data/fred_oecd/Commodities'

comms_raw = read_all_csvs_in_folder(folder_path)


for i in range(len(comms_raw)):
    
    comms_raw[i]['DATE'] = pd.to_datetime(comms_raw[i]['DATE'])
    comms_raw[i].rename(columns={'DATE': 'Date'}, inplace=True)  
    comms_raw[i]['Date'] = comms_raw[i]['Date'].dt.to_period('M')


# Merge the dataframes on the 'Date' column
comms = comms_raw[0]

for i in range(1, len(comms_raw)):
    comms = pd.merge(comms, comms_raw[i], on='Date', how='outer')

comms.sort_values(by='Date', inplace=True)
comms.ffill(inplace=True)


# ## Joining with Accounting Data

acc_items_clean.drop_duplicates(subset=['tic', 'Date'], inplace=True)

# Group dataframe by 'tic' column
grouped = acc_items_clean.groupby('tic')

# Apply the resample group function to each group
acc_items_resampled = grouped.apply(resample_group)

# Reset the index
acc_items_resampled.reset_index(drop=True, inplace=True)

acc_items_resampled[acc_items_resampled['tic'] == 'AAPL']

merged = pd.merge(acc_items_resampled, macros, on='Date', how='outer')
merged = pd.merge(merged, energy, on='Date', how='outer')
merged = pd.merge(merged, comms, on='Date', how='outer')



# ## Company Information


folder_path = 'data/refinitiv/comp_info'

comp_info_raw = read_all_csvs_in_folder(folder_path)


esg = comp_info_raw[0]
employees = comp_info_raw[1]
industry = comp_info_raw[2]
free_float = comp_info_raw[3]


# ### ESG Score

esg['Date'] = pd.to_datetime(esg['Date']).dt.to_period('M')
esg.drop_duplicates(subset=['RIC', 'Date'], inplace=True)
esg.drop(columns='Instrument', inplace=True)
esg.rename(columns={'RIC': 'ric'}, inplace=True)


merged = pd.merge(merged, esg, on=['Date', 'ric'], how='left')
merged.sort_values(by=['tic', 'Date'], inplace=True, na_position='first')


# ### Number of Employees


employees['Date'] = pd.to_datetime(employees['Number Employees Date']).dt.to_period('M')
employees.drop_duplicates(subset=['RIC', 'Date'], inplace=True)
employees.drop(columns=['Instrument', 'Number Employees Date'], inplace=True)
employees.rename(columns={'RIC': 'ric'}, inplace=True)


merged = pd.merge(merged, employees, on=['Date', 'ric'], how='left') # should I use 'left'? with 'outer' and ffill() the first year employee values could in theory be preserved
merged.sort_values(by=['tic', 'Date'], inplace=True, na_position='first')


# ### Free Float

free_float['Date'] = pd.to_datetime(free_float['Date']).dt.to_period('M')
free_float.drop_duplicates(subset=['RIC', 'Date'], inplace=True)
free_float.drop(columns='Instrument', inplace=True)
free_float.rename(columns={'RIC': 'ric'}, inplace=True)



merged = pd.merge(merged, free_float, on=['Date', 'ric'], how='left')
merged.sort_values(by=['tic', 'Date'], inplace=True, na_position='first')



# ### Industry Classification

print(f"Number of unique TRBC Economic Sectors: {industry['TRBC Economic Sector Name'].nunique()}")
print(f"Number of unique TRBC Business Sectors: {industry['TRBC Business Sector Name'].nunique()}")
print(f"Number of unique TRBC Industry Groups:  {industry['TRBC Industry Group Name'].nunique()}")
print(f"Number of unique TRBC Industries:       {industry['TRBC Industry Name'].nunique()}")


industry.drop(columns=['Instrument', 'TRBC Industry Group Name', 'TRBC Industry Name'], inplace=True)
industry.rename(columns={'RIC': 'ric'}, inplace=True)


merged = pd.merge(merged, industry, on='ric', how='left')



merged['month'] = merged['Date'].dt.month


# ### Clean


data_full = merged.copy()

# Replace 0s with NAs in 'Number of Employees'
data_full['Number of Employees'] = data_full['Number of Employees'].replace(0, np.nan)

# Backward fill ESG score with the assumption that the first ESG score is a rough estimate for the company in previous years
data_full['ESG Score'] = data_full.groupby('ric', group_keys=False)['ESG Score'].bfill()

# Forward fill each value to add missing monthly values for data that is only published quarterly/yearly
data_full = data_full.groupby('ric', group_keys=False).apply(lambda group: group.ffill())

# Backward fill 'Number of Employees' with the assumption that the first number is a rough estimate for the previous years
data_full['Number of Employees'] = data_full.groupby('ric')['Number of Employees'].bfill()

data_full[data_full['tic'] == 'AAPL']


# ## Target

# ### Import and Format

marketcaps = pd.read_csv('data/marketcaps_Jan2000_May2023.csv', index_col=0, parse_dates=[2])
marketcaps['Date'] = marketcaps['Date'].dt.to_period('M')
marketcaps.rename(columns={'RIC': 'ric'}, inplace=True)
marketcaps.drop(columns='Instrument', inplace=True)


# ### Merging and Preparing

data_incl_target = pd.merge(data_full, marketcaps, on=['ric', 'Date'], how='left')

data_cleaned = data_incl_target.drop(columns=['fic', 'Constituent Name', 'ric'])
# data_cleaned.set_index(['tic', 'Date'], inplace=True)


# 66 raw features, two of them are for the industry classification and will be transformed into dummy variables (one-hot encoding)


# List of columns not to shift
cols_to_exclude = ['Date', 'tic', 'Company Market Cap', 'month']

# Columns to shift
cols_to_shift = [col for col in data_cleaned.columns if col not in cols_to_exclude]

# Group dataframe by 'tic' column and apply the function
data_cleaned_shifted = data_cleaned.groupby('tic', group_keys=False).apply(shift_group)

# Drop all rows that contain missing values
data_cleaned_shifted.dropna(inplace=True)


data_cleaned_shifted.set_index(['tic', 'Date'], inplace=True)

# Rename the columns that have unmeaningful names due to OECD download
data_cleaned_shifted.rename(columns={'Value_x': 'CCI', 'Value_y': 'BCI'}, inplace=True)

# Save CSV
data_cleaned_shifted.to_csv('data/cleaned_dataset_2M_delay.csv')

#------------------------------------------------------------------------------
## Create next return DF for evaluation
next_return_df = data_cleaned_shifted.reset_index()[['tic', 'Date', 'Company Market Cap']]

next_return_df = next_return_df.groupby('tic', group_keys=False).apply(calculate_next_1M_return)
next_return_df = next_return_df.groupby('tic', group_keys=False).apply(calculate_next_6M_return)
next_return_df = next_return_df.groupby('tic', group_keys=False).apply(calculate_next_12M_return)


next_return_df.drop(columns='Company Market Cap').to_csv('data/next_returns.csv')


#------------------------------------------------------------------------------
# Get S&P500 returns from Yahoo finance
sp500_returns = yf.download('^GSPC', start='2000-01-01', end='2023-07-09', interval='1mo')
sp500_returns = sp500_returns.reset_index()
sp500_returns['Date'] = sp500_returns['Date'].dt.to_period('M')
sp500_returns['Next 1M Adj Close'] = sp500_returns['Adj Close'].shift(-1)
sp500_returns['Next 6M Adj Close'] = sp500_returns['Adj Close'].shift(-6)
sp500_returns['Next 12M Adj Close'] = sp500_returns['Adj Close'].shift(-12)

sp500_returns['Next 1M Index Return (%)'] = ((sp500_returns['Next 1M Adj Close']/sp500_returns['Adj Close'])-1)*100
sp500_returns['Next 6M Index Return (%)'] = ((sp500_returns['Next 6M Adj Close']/sp500_returns['Adj Close'])-1)*100
sp500_returns['Next 12M Index Return (%)'] = ((sp500_returns['Next 12M Adj Close']/sp500_returns['Adj Close'])-1)*100

sp500_returns = sp500_returns[['Date', 'Adj Close', 'Next 1M Index Return (%)', 'Next 6M Index Return (%)', 'Next 12M Index Return (%)']]
sp500_returns.to_csv('data/next_index_returns.csv')
