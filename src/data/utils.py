from scipy.stats import normaltest

def is_normal(feature_series, alpha=0.01):
    
    '''
    Returns boolean value, indicating whether a feature is normally distributed or not. 
    '''
    
    k2, p = normaltest(feature_series)
    if p < alpha:
        return False
    else:
        return True