from sklearn.decomposition import PCA


def apply_pca(Xtrn_full=None, Xtrn=None, Xval=None, Xtst=None, expl_var=0.9, val_set=True):
    """
    Applies Principal Component Analysis (PCA) transformation to the provided datasets.
    PCA is fitted on the training data and then used to transform the training, validation, and test sets.
    The function also prints the number of principal components and the explained variance of the first three components.
    
    Args:
        Xtrn_full (numpy.ndarray, optional): The full training data. This parameter is used when there is no validation set.
        Xtrn (numpy.ndarray, optional): The training data. This parameter is used when there is a validation set.
        Xval (numpy.ndarray, optional): The validation data. This parameter is used when there is a validation set.
        Xtst (numpy.ndarray): The test data.
        expl_var (float, optional): The desired level of explained variance. PCA components are selected so that they explain this level of total variance. Default is 0.9.
        val_set (bool, optional): A flag indicating whether a validation set is provided. Default is True.
    
    Raises:
        ValueError: If the appropriate combination of Xtrn_full, Xtrn, Xval, and Xtst is not provided.
        
    Returns:
        tuple: A tuple containing:
            - The transformed training data (numpy.ndarray)
            - The transformed validation data if val_set is True (numpy.ndarray)
            - The transformed test data (numpy.ndarray)
            - The number of principal components (int)
    """
    
    expl_var = expl_var
    
    if val_set:
        
        if Xtrn is None or Xval is None or Xtst is None:
            raise ValueError("Please provide Xtrn, Xval and Xtst ")
        
        # Define and fit pca
        pca = PCA(n_components=expl_var).fit(Xtrn)
        
        Xtrn_pca = pca.transform(Xtrn)
        Xval_pca = pca.transform(Xval)
        Xtst_pca = pca.transform(Xtst)
        
        
    else:
        
        if Xtrn_full is None or Xtst is None:
            raise ValueError("Please provide Xtrn_full and Xtst ")
            
        # Define and fit pca
        pca = PCA(n_components=expl_var).fit(Xtrn_full)
        
        Xtrn_full_pca = pca.transform(Xtrn_full)
        Xtst_pca = pca.transform(Xtst)
       
       
    nPCs = pca.components_.shape[0]
    print(f"The number of principal components that explain {expl_var*100}% of the total variance is {pca.components_.shape[0]}.")
    print(f"The first component explains approximately {pca.explained_variance_ratio_[0]*100:.2f}% of the total variance.")
    print(f"The second component explains approximately {pca.explained_variance_ratio_[1]*100:.2f}% of the total variance.")
    print(f"The third component explains approximately {pca.explained_variance_ratio_[2]*100:.2f}% of the total variance.")
        
    if val_set:
        return Xtrn_pca, Xval_pca, Xtst_pca, nPCs
    
    else:
        return Xtrn_full_pca, Xtst_pca, nPCs