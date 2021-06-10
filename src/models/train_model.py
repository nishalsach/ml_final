import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
import time

def evaluate_models(regression_model, train_X, train_y, n_folds=5):
    
    '''
    Input:
    
    regression_model: Model to conduct regression
    train_X: Features from the training data
    train_y: Targets from the training data
    
    Function sets up a pipeline, custom_scorers, and conducts 
    a GridSearch for the best parameters. It returns the scores from the 
    estimated best parameters for the chosen model. 
    
    '''
    
    # Initialise scalers and scale
    # scaler = RobustScaler().fit(train_X)
    scaler = MinMaxScaler()
    
    # Setup pipeline
    pipe = make_pipeline(
        scaler, # The scaler
        regression_model # The model
    )
    
    # Setup the custom scorers
    custom_scorers = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2_score': 'r2',
        }
    
    print(f"Calling {n_folds}-fold Cross-Validation ... ")
    
    # Start time
    start = time.time()
    
    # Setup custom scorers
    scores = cross_validate(
        pipe, 
        train_X, 
        train_y, 
        cv=n_folds, 
        scoring = custom_scorers, 
        n_jobs = -1, 
        verbose=0
        
    )
    
    # Time taken
    time_taken = time.time() - start
    
    print(f"Training complete, time taken: {time_taken}\n")
    
    return scores

def select_k_best(train_X, mir_array, k):
    
    '''
    
    Input: 
    train_X: DataFrame of features for the train data
    mir_array: array of mutual importances for each feature with train_y
    
    Output:
    train_X_k: train_X with the k-best features chosen, based on mutual importance values
    
    '''
    
    # Get indices of best k features
    k_best_indices = (-mir_array).argsort()[:k]
    
    # Subset train_X by best k features
    train_X_k = train_X.iloc[:, k_best_indices]
    
    return train_X_k