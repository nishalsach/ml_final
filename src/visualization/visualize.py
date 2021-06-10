import matplotlib.pyplot as plt

def plot_model_evals(mean_df, std_df, k_value):

    mean_df.plot(
        kind='bar', 
        subplots=True, 
        layout=(1, 3), 
        figsize=(14, 8), 
        rot=45, 
        yerr = std_df, 
        grid=True, 
        sharex=False, 
        xlabel = 'Model Name', 
        ylabel = 'Metric Value', 
        legend=False, 
        alpha = 0.9, 
        title = f'Comparing Model Performance (5-Fold CV, {k_value} Features)', 
        color=['red', 'orange', 'dodgerblue']
    );

    plt.tight_layout()