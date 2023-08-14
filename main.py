# Import custom modules
import PrePro
import DimReduction
import Visual
import EvalFunctions as ev
import EstAlpha
import MultipleValuation
import BGLR
import LR
import RF
import ELM
import MLP
import Ensemble


#______________________________________________________________________________
# 1. IMPORT DATA AND PREPROCESSING
# For the preparation of the saved datasets that are loaded here, see DataPrep.py

dataset = PrePro.read_clean_dataset()
features, target = PrePro.prepare_features_and_target(dataset)

Xtrn_full, Xtrn, Xval, Xtst, Ytrn_full, Ytrn, Yval, Ytst = \
    PrePro.train_val_test_split(features, target)

Xtrn_full_minmax, Xtrn_minmax, Xval_minmax, Xtst_minmax = \
    PrePro.scale_minmax(Xtrn_full, Xtrn, Xval, Xtst)
    
Xtrn_full_minmax_win, Xtrn_minmax_win, Xval_minmax_win, Xtst_minmax_win = \
    PrePro.winsorize_and_scale(Xtrn_full, Xtrn, Xval, Xtst)
    
target_list_log = PrePro.log_transform_target(Ytrn_full, Ytrn, Yval, Ytst)
Visual.visualise_log_transform(target)

Xtrn_minmax_pca, Xval_minmax_pca, Xtst_minmax_pca, nPCs = DimReduction.apply_pca(
    Xtrn=Xtrn_minmax, Xval=Xval_minmax, Xtst=Xtst_minmax, val_set=True)

#______________________________________________________________________________
# 2. BASELINE VALUATIONS

# 2(1) RUNNING BASELINE MODELS

# Run Multiples Valuation
multiples_df = MultipleValuation.create_and_save_multiple_data()
ev1_multiples = ev.evaluate_quintile_returns_multiples(
    multiples_df, save_eval_df=True, name="PE Multiples Valuation")

# Run Bartram and Grinblatt (2018) Linear Regression (BGLR)
BGLR.apply_BGLR(features, target, save_eval_df=True, scaling=True, name="BGLR (with scaling)")
BGLR.apply_BGLR(features, target, save_eval_df=True, scaling=False, name="BGLR (no scaling)")

# Run OLS regression
LR.apply_LR(Xtrn_full_minmax, Ytrn_full, Xtst_minmax, Ytst,
            log_transformed=False, save_eval_df=True, name="LR (no log)")

LR.apply_LR(Xtrn_full_minmax, target_list_log[0], Xtst_minmax, Ytst, # use no log test target
            log_transformed=True, save_eval_df=True, name="LR (yes log)")


# 2(2) ESTIMATING BASELINE ALPHA
baselines = ["PE Multiples Valuation", "BGLR (with scaling)", "BGLR (no scaling)",
             "LR (no log)", "LR (yes log)"]

periods = ['1M', '6M', '12M']

# Estimating abnormal return (alpha) for baseline models
for model_name in baselines:
    for p in periods:
        # Quintile portfolio level returns not explained by index return 
        EstAlpha.run_portfolio_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Quintile portfolio level returns not explained by FF5FM
        EstAlpha.run_portfolio_lvl_regression_FF5FM(model_name, period=p, save=True)
        
        # Stock level returns not explained by index return 
        EstAlpha.run_stock_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Stock level returns not explained by FF5FM
        EstAlpha.run_stock_lvl_regression_FF5FM(model_name, period=p, save=True)
        
# Create summary Excel files
EstAlpha.est_alphas_summary(baselines, factors='CAPM', lvl='QPortfolio', save_as="baselines summary_CAPM_Q")
EstAlpha.est_alphas_summary(baselines, factors='FF5FM', lvl='QPortfolio', save_as="baselines summary_FF_Q")
EstAlpha.est_alphas_summary(baselines, factors='CAPM', lvl='stock', save_as="baselines summary_CAPM_S")
EstAlpha.est_alphas_summary(baselines, factors='FF5FM', lvl='stock', save_as="baselines summary_FF_S")


#______________________________________________________________________________
# 3. MACHINE LEARNING (ML) VALUATIONS


# (i) Random Forest (RF)
RF.apply_RF(Xtrn_full_minmax, Ytrn_full, Xtst_minmax, Ytst,
            log_transformed=False, pca=False, save_eval_df=True, name="RF (no log, no pca)")

RF.apply_RF(Xtrn_full_minmax, target_list_log[0], Xtst_minmax, Ytst,
            log_transformed=True, pca=False, save_eval_df=True, name="RF (yes log, no pca)")

RF.apply_RF(Xtrn_full_minmax, Ytrn_full, Xtst_minmax, Ytst,
            log_transformed=False, pca=True, save_eval_df=True, name="RF (no log, yes pca)")

RF.apply_RF(Xtrn_full_minmax, target_list_log[0], Xtst_minmax, Ytst,
            log_transformed=True, pca=True, save_eval_df=True, name="RF (yes log, yes pca)")

RFs = ["RF (no log, no pca)", "RF (yes log, no pca)",
       "RF (no log, yes pca)", "RF (yes log, yes pca)"]

# Estimating abnormal return (alpha) for random forest models
for model_name in RFs:
    for p in periods:
        # Quintile portfolio level returns not explained by index return 
        EstAlpha.run_portfolio_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Quintile portfolio level returns not explained by FF5FM
        EstAlpha.run_portfolio_lvl_regression_FF5FM(model_name, period=p, save=True)
        
        # Stock level returns not explained by index return 
        EstAlpha.run_stock_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Stock level returns not explained by FF5FM
        EstAlpha.run_stock_lvl_regression_FF5FM(model_name, period=p, save=True)

# Create summary Excel files
EstAlpha.est_alphas_summary(RFs, factors='CAPM', lvl='QPortfolio', save_as="RFs summary_CAPM_Q")
EstAlpha.est_alphas_summary(RFs, factors='FF5FM', lvl='QPortfolio', save_as="RFs summary_FF_Q")
EstAlpha.est_alphas_summary(RFs, factors='CAPM', lvl='stock', save_as="RFs summary_CAPM_S")
EstAlpha.est_alphas_summary(RFs, factors='FF5FM', lvl='stock', save_as="RFs summary_FF_S")


# (ii) Extreme Learning Machine (ELM)
best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_minmax, Ytrn, Xval_minmax, Yval)
ev.evaluate_quintile_returns(
    best_elm, Xtst_minmax, Ytst, log_transformed=False,
    save_eval_df=True, name="ELM (no log, no pca)")

best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_minmax, target_list_log[1], Xval_minmax, target_list_log[2])
ev.evaluate_quintile_returns(
    best_elm, Xtst_minmax, Ytst, log_transformed=True,
    save_eval_df=True, name="ELM (yes log, no pca)")

best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_minmax_pca, Ytrn, Xval_minmax_pca, Yval)
ev.evaluate_quintile_returns(
    best_elm, Xtst_minmax_pca, Ytst, log_transformed=False,
    save_eval_df=True, name="ELM (no log, yes pca)")

best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_minmax_pca, target_list_log[1], Xval_minmax_pca, target_list_log[2])
ev.evaluate_quintile_returns(
    best_elm, Xtst_minmax_pca, Ytst, log_transformed=True,
    save_eval_df=True, name="ELM (yes log, yes pca)")


ELMs = ["ELM (no log, no pca)", "ELM (yes log, no pca)",
       "ELM (no log, yes pca)", "ELM (yes log, yes pca)"]

# Estimating abnormal return (alpha) for ELMs
for model_name in ELMs:
    for p in periods:
        # Quintile portfolio level returns not explained by index return 
        EstAlpha.run_portfolio_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Quintile portfolio level returns not explained by FF5FM
        EstAlpha.run_portfolio_lvl_regression_FF5FM(model_name, period=p, save=True)
        
        # Stock level returns not explained by index return 
        EstAlpha.run_stock_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Stock level returns not explained by FF5FM
        EstAlpha.run_stock_lvl_regression_FF5FM(model_name, period=p, save=True)

# Create summary Excel files
EstAlpha.est_alphas_summary(ELMs, factors='CAPM', lvl='QPortfolio', save_as="EMLs summary_CAPM_Q")
EstAlpha.est_alphas_summary(ELMs, factors='FF5FM', lvl='QPortfolio', save_as="EMLs summary_FF_Q")
EstAlpha.est_alphas_summary(ELMs, factors='CAPM', lvl='stock', save_as="EMLs summary_CAPM_S")
EstAlpha.est_alphas_summary(ELMs, factors='FF5FM', lvl='stock', save_as="EMLs summary_FF_S")


# (iii) Multilayer Perceptron (MLP)

# (a)

# Run topological grid search 3-fold cross validation on valuation accuracy
best_mlp, best_mae, best_neurons, best_n_layers = MLP.topological_grid_search_CV_MLP(
    Xtrn_full_minmax, Ytrn_full, 
    hidden_layers = [1, 2, 3], neurons = [16, 32, 64, 128, 512])

# -> best accuracy achieved with the most complex model: 3 layers and 512 neurons

# Train this deep neural net with an early stopping mechanism to avoid overfitting
MLP.apply_MLP_3HL(Xtrn_minmax, Ytrn, Xval_minmax, Yval,
                       Xtst_minmax, Ytst, neurons=512,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL512N (no log, no pca)")

MLP.apply_MLP_3HL(Xtrn_minmax, target_list_log[1], Xval_minmax, 
                       target_list_log[2], Xtst_minmax, Ytst, neurons=512,
                       log_transformed=True, save_eval_df=True, 
                       name="NN 3HL512N (yes log, no pca)")

MLP.apply_MLP_3HL(Xtrn_minmax_pca, Ytrn, Xval_minmax_pca, Yval,
                       Xtst_minmax_pca, Ytst, neurons=512,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL512N (no log, yes pca)")

MLP.apply_MLP_3HL(Xtrn_minmax_pca, target_list_log[1], Xval_minmax_pca, 
                       target_list_log[2], Xtst_minmax_pca, Ytst, neurons=512,
                       log_transformed=True, save_eval_df=True, 
                       name="NN 3HL512N (yes log, yes pca)")


MLP_3HL512N_list = ["NN 3HL512N (no log, no pca)", "NN 3HL512N (yes log, no pca)",
                    "NN 3HL512N (no log, yes pca)", "NN 3HL512N (yes log, yes pca)"]

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(MLP_3HL512N_list, factors='CAPM', lvl='QPortfolio', save_as="MLP_3HL512N summary_CAPM_Q")
EstAlpha.est_alphas_summary(MLP_3HL512N_list, factors='FF5FM', lvl='QPortfolio', save_as="MLP_3HL512N summary_FF_Q")
EstAlpha.est_alphas_summary(MLP_3HL512N_list, factors='CAPM', lvl='stock', save_as="MLP_3HL512N summary_CAPM_S")
EstAlpha.est_alphas_summary(MLP_3HL512N_list, factors='FF5FM', lvl='stock', save_as="MLP_3HL512N summary_FF_S")

# =============================================================================
# Note that both grid searches were done for the minmax scaled values but without  
# reduced dimensionality after pca transformation or a log transformed target.
# This implies that the optimal topology might be different for a grid search
# with these changes.

# So far, results do not look too convincing, let's try topological grid search but the 
# target being the quintile return spread and not high accuracy on valuation set
# =============================================================================


# (b)
# second topological grid search with average monthly quintile spread return as target (similar to ELM grid)
best_mlp, best_performance, best_neurons, best_n_layers = MLP.topo_grid_MLP_2(
    Xtrn_minmax, Ytrn, Xval_minmax, Yval)

# -> best quintile spread achieved with 3 layers and 64 neurons


# Train this deep neural net with an early stopping mechanism to avoid overfitting
MLP.apply_MLP_3HL(Xtrn_minmax, Ytrn, Xval_minmax, Yval,
                       Xtst_minmax, Ytst, neurons=64,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL64N (no log, no pca)")

MLP.apply_MLP_3HL(Xtrn_minmax, target_list_log[1], Xval_minmax, 
                       target_list_log[2], Xtst_minmax, Ytst, neurons=64,
                       log_transformed=True, save_eval_df=True, 
                       name="NN 3HL64N (yes log, no pca)")

MLP.apply_MLP_3HL(Xtrn_minmax_pca, Ytrn, Xval_minmax_pca, Yval,
                       Xtst_minmax_pca, Ytst, neurons=64,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL64N (no log, yes pca)")

MLP.apply_MLP_3HL(Xtrn_minmax_pca, target_list_log[1], Xval_minmax_pca, 
                       target_list_log[2], Xtst_minmax_pca, Ytst, neurons=64,
                       log_transformed=True, save_eval_df=True, 
                       name="NN 3HL64N (yes log, yes pca)")

MLP_3HL64N_list = ["NN 3HL64N (no log, no pca)", "NN 3HL64N (yes log, no pca)",
                    "NN 3HL64N (no log, yes pca)", "NN 3HL64N (yes log, yes pca)"]

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(MLP_3HL64N_list, factors='CAPM', lvl='QPortfolio', save_as="MLP_3HL64N summary_CAPM_Q")
EstAlpha.est_alphas_summary(MLP_3HL64N_list, factors='FF5FM', lvl='QPortfolio', save_as="MLP_3HL64N summary_FF_Q")
EstAlpha.est_alphas_summary(MLP_3HL64N_list, factors='CAPM', lvl='stock', save_as="MLP_3HL64N summary_CAPM_S")
EstAlpha.est_alphas_summary(MLP_3HL64N_list, factors='FF5FM', lvl='stock', save_as="MLP_3HL64N summary_FF_S")


#______________________________________________________________________________
# Evaluate all current ML models

all_ML_models = RFs + ELMs + MLP_3HL512N_list + MLP_3HL64N_list

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(all_ML_models, factors='CAPM', lvl='QPortfolio', save_as="MLs_1 summary_CAPM_Q")
EstAlpha.est_alphas_summary(all_ML_models, factors='FF5FM', lvl='QPortfolio', save_as="MLs_1 summary_FF_Q")
EstAlpha.est_alphas_summary(all_ML_models, factors='CAPM', lvl='stock', save_as="MLs_1 summary_CAPM_S")
EstAlpha.est_alphas_summary(all_ML_models, factors='FF5FM', lvl='stock', save_as="MLs_1 summary_FF_S")



#______________________________________________________________________________
# Create raw return summary tables for the best models and baselines

ev.create_xlsx_ev3(["RF (no log, yes pca)", "ELM (no log, yes pca)",
                    "ELM (yes log, yes pca)", "NN 3HL64N (no log, yes pca)"])

ev.create_xlsx_ev3(["PE Multiples Valuation", "BGLR (no scaling)", 
                    "LR (no log)", "LR (yes log)"])

#______________________________________________________________________________
# 4. ENSEMBLE

# Ensemble averaging the predictions of the best RF, ELM, and MLP
Ensemble.apply_and_evaluate_ensemble()

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(["Ensemble"], factors='CAPM', lvl='QPortfolio', save_as="Ensemble summary_CAPM_Q")
EstAlpha.est_alphas_summary(["Ensemble"], factors='FF5FM', lvl='QPortfolio', save_as="Ensemble summary_FF_Q")
EstAlpha.est_alphas_summary(["Ensemble"], factors='CAPM', lvl='stock', save_as="Ensemble summary_CAPM_S")
EstAlpha.est_alphas_summary(["Ensemble"], factors='FF5FM', lvl='stock', save_as="Ensemble summary_FF_S")


#______________________________________________________________________________
# 5. BEST ML MODELS WITH LESS DATA


# Select 30 accounting items only (27 BG balance sheet/income statement and 3 cashflow from Refinitiv)
Xtrn_full_minmax_acc = Xtrn_full_minmax[:, :30]
Xtrn_minmax_acc = Xtrn_minmax[:, :30]
Xval_minmax_acc = Xval_minmax[:, :30]
Xtst_minmax_acc = Xtst_minmax[:, :30]

Xtrn_minmax_pca_acc, Xval_minmax_pca_acc, Xtst_minmax_pca_acc, nPCs_acc = DimReduction.apply_pca(
    Xtrn=Xtrn_minmax_acc, Xval=Xval_minmax_acc, Xtst=Xtst_minmax_acc, val_set=True)


# Run best ML model configurations from above on less data
RF.apply_RF(Xtrn_full_minmax_acc, Ytrn_full, Xtst_minmax_acc, Ytst,
            log_transformed=False, pca=True, save_eval_df=True, name="RF (pca) - less data")


best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_minmax_pca_acc, Ytrn, Xval_minmax_pca_acc, Yval)
ev.evaluate_quintile_returns(
    best_elm, Xtst_minmax_pca_acc, Ytst, log_transformed=False,
    save_eval_df=True, name="ELM (pca) - less data")


MLP.apply_MLP_3HL(Xtrn_minmax_pca_acc, Ytrn, Xval_minmax_pca_acc, Yval,
                       Xtst_minmax_pca_acc, Ytst, neurons=64,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL64N (pca) - less data")


reduced_data_models = ["RF (pca) - less data", "ELM (pca) - less data", "NN 3HL64N (pca) - less data"]

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(reduced_data_models, factors='CAPM', lvl='QPortfolio', save_as="Acc_only summary_CAPM_Q")
EstAlpha.est_alphas_summary(reduced_data_models, factors='FF5FM', lvl='QPortfolio', save_as="Acc_only summary_FF_Q")
EstAlpha.est_alphas_summary(reduced_data_models, factors='CAPM', lvl='stock', save_as="Acc_only summary_CAPM_S")
EstAlpha.est_alphas_summary(reduced_data_models, factors='FF5FM', lvl='stock', save_as="Acc_only summary_FF_S")


# Calculate percentage change and create xlsx summary
EstAlpha.calculate_percentage_change_xlsx(["RF (no log, yes pca)", "ELM (no log, yes pca)",
                                           "NN 3HL64N (no log, yes pca)"],
                                          reduced_data_models, save_as="best_ML")
    















#______________________________________________________________________________
###############################################################################
# Appendix (i): Saving detailed alpha regression results for MLPs, Ensemble, and less data models

remaining_models = MLP_3HL512N_list + MLP_3HL64N_list + ["Ensemble"] + reduced_data_models

for model_name in remaining_models:
    for p in periods:
        # Quintile portfolio level returns not explained by index return 
        EstAlpha.run_portfolio_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Quintile portfolio level returns not explained by FF5FM
        EstAlpha.run_portfolio_lvl_regression_FF5FM(model_name, period=p, save=True)
        
        # Stock level returns not explained by index return 
        EstAlpha.run_stock_lvl_regression_CAPM(model_name, period=p, save=True)
        
        # Stock level returns not explained by FF5FM
        EstAlpha.run_stock_lvl_regression_FF5FM(model_name, period=p, save=True)




#______________________________________________________________________________
# Appendix (ii): Visualising winsorization and testing best models on winsorized data
Visual.visualise_winsorizations(Xtrn_full_minmax_win)

Xtrn_minmax_pca_win, Xval_minmax_pca_win, Xtst_minmax_pca_win, nPCs_win = DimReduction.apply_pca(
    Xtrn=Xtrn_minmax_win, Xval=Xval_minmax_win, Xtst=Xtst_minmax_win, val_set=True)


RF.apply_RF(Xtrn_full_minmax_win, Ytrn_full, Xtst_minmax_win, Ytst,
            log_transformed=False, pca=True, save_eval_df=True, name="RF (pca) - winsor")


best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_minmax_pca_win, Ytrn, Xval_minmax_pca_win, Yval)
ev.evaluate_quintile_returns(
    best_elm, Xtst_minmax_pca_win, Ytst, log_transformed=False,
    save_eval_df=True, name="ELM (pca) - winsor")


MLP.apply_MLP_3HL(Xtrn_minmax_pca_win, Ytrn, Xval_minmax_pca_win, Yval,
                       Xtst_minmax_pca_win, Ytst, neurons=64,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL64N (pca) - winsor")


winsor_data_models = ["RF (pca) - winsor", "ELM (pca) - winsor", "NN 3HL64N (pca) - winsor"]

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(winsor_data_models, factors='CAPM', lvl='QPortfolio', save_as="winsor summary_CAPM_Q")
EstAlpha.est_alphas_summary(winsor_data_models, factors='FF5FM', lvl='QPortfolio', save_as="winsor summary_FF_Q")
EstAlpha.est_alphas_summary(winsor_data_models, factors='CAPM', lvl='stock', save_as="winsor summary_CAPM_S")
EstAlpha.est_alphas_summary(winsor_data_models, factors='FF5FM', lvl='stock', save_as="winsor summary_FF_S")


# -> no improvement when winsorizing outliers


#______________________________________________________________________________
# Appendix (iii): Testing standard scaling instead of minmax on best models
Xtrn_full_stand, Xtrn_stand, Xval_stand, Xtst_stand = \
    PrePro.scale_standard(Xtrn_full, Xtrn, Xval, Xtst)
    
Xtrn_stand_pca, Xval_stand_pca, Xtst_stand_pca, nPCs_stand = DimReduction.apply_pca(
    Xtrn=Xtrn_stand, Xval=Xval_stand, Xtst=Xtst_stand, val_set=True)



RF.apply_RF(Xtrn_full_stand, Ytrn_full, Xtst_stand, Ytst,
            log_transformed=False, pca=True, save_eval_df=True, name="RF (pca) - standard_scaler")


best_elm, best_neurons, best_performance = ELM.grid_search_elm(
    Xtrn_stand_pca, Ytrn, Xval_stand_pca, Yval)
ev.evaluate_quintile_returns(
    best_elm, Xtst_stand_pca, Ytst, log_transformed=False,
    save_eval_df=True, name="ELM (pca) - standard_scaler")


MLP.apply_MLP_3HL(Xtrn_stand_pca, Ytrn, Xval_stand_pca, Yval,
                       Xtst_stand_pca, Ytst, neurons=64,
                       log_transformed=False, save_eval_df=True, 
                       name="NN 3HL64N (pca) - standard_scaler")


standard_scaler_models = ["RF (pca) - standard_scaler", "ELM (pca) - standard_scaler", "NN 3HL64N (pca) - standard_scaler"]

# Create alpha summary Excel files
EstAlpha.est_alphas_summary(standard_scaler_models, factors='CAPM', lvl='QPortfolio', save_as="standard_scaler summary_CAPM_Q")
EstAlpha.est_alphas_summary(standard_scaler_models, factors='FF5FM', lvl='QPortfolio', save_as="standard_scaler summary_FF_Q")
EstAlpha.est_alphas_summary(standard_scaler_models, factors='CAPM', lvl='stock', save_as="standard_scaler summary_CAPM_S")
EstAlpha.est_alphas_summary(standard_scaler_models, factors='FF5FM', lvl='stock', save_as="standard_scaler summary_FF_S")



# -> no improvement when using standard scaling






