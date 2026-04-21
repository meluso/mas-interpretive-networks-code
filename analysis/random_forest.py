# analysis/random_forest.py
"""Random Forest analysis for AI Teams task-network interactions.

Uses permutation importance with confidence intervals to identify important features
and their interactions without requiring computationally expensive SHAP analysis.

Key implementation features:
- Uses permutation-based feature importance for reliable ranking of correlated features
- Calculates confidence intervals through repeated permutations (n_repeats=15)
- Identifies important interactions between network properties and task dimensions
- Generates combined interaction lists for subsequent ridge regression analysis
- Supports memory-optimized processing for large datasets (using max_samples parameter)
"""

from itertools import combinations
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder
import time

# Import constants and utility functions
from analysis.util import (
    get_control_variables,
    get_network_metrics, 
    get_task_difficulties,
    get_outcome_metrics,
    get_categorical_variables,
    load_parquet_dataset
)

# Set global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def preprocess_random_forest(df, categorical_features, continuous_features):
    """Preprocess data for random forest analysis using already transformed features."""
    # Create a copy of the DataFrame with selected features
    X = df[categorical_features + continuous_features].copy()
    
    # Create preprocessing pipeline for just encoding categorical features
    # Continuous features are already normalized in the dataset
    if categorical_features:
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            # No scaling needed for continuous features as they're already normalized
            ('passthrough', 'passthrough', continuous_features)
        ], verbose_feature_names_out=False)
        
        X_transformed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
    else:
        # If no categorical features, just use the continuous features directly
        X_transformed = X.values
        feature_names = X.columns.tolist()
    
    return preprocessor if categorical_features else None, X_transformed, feature_names


def partition_data(df, train_frac=0.8):
    """Partition data into training and test sets using trial_id."""
    print("  Starting data partitioning...")
    
    # Get the number of trials dynamically
    num_trials = df['trial_id'].nunique()
    
    # Calculate training cutoff (80% of trials)
    train_size = int(num_trials * train_frac)
    
    # Direct partitioning using trial_id ranges
    train_indices = df[df['trial_id'] < train_size].index.tolist()
    test_indices = df[df['trial_id'] >= train_size].index.tolist()
    
    n_total = len(df)
    n_train = len(train_indices)
    n_test = len(test_indices)
    
    print(f"  Data partitioned successfully:")
    print(f"    Training set: {n_train} samples ({n_train/n_total*100:.1f}%)")
    print(f"    Test set: {n_test} samples ({n_test/n_total*100:.1f}%)")
    
    return train_indices, test_indices

def train_random_forest(X_train, y_train, n_estimators=200):
    """Train random forest model."""
    print(f"  Training Random Forest model with {n_estimators} trees...")
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=cpu_count()-1,
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"  Model training completed in {train_time:.1f} seconds")
    
    return model

def calculate_permutation_importance_with_ci(model, X_test, y_test, feature_names, n_repeats=15, max_samples=0.01):
    """Calculate permutation importance with confidence intervals."""
    print(f"  Calculating permutation importance with {n_repeats} repeats, {int(max_samples*100)}% of test data,...")
    start_time = time.time()
    
    # Memory-optimized settings
    result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=1,
        max_samples=max_samples
    )
    
    # Calculate 95% confidence intervals
    importances = result.importances
    lower_bounds, upper_bounds = np.percentile(importances, [2.5, 97.5], axis=1)
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Importance_Std': result.importances_std,
        'Lower_CI': lower_bounds,
        'Upper_CI': upper_bounds
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Identify significant features (lower bound of CI > 0)
    importance_df['Significant'] = importance_df['Lower_CI'] > 0
    
    importance_time = time.time() - start_time
    print(f"  Permutation importance calculated in {importance_time:.1f} seconds")
    print(f"  Found {importance_df['Significant'].sum()} significant features")
    
    return importance_df

def generate_pairwise_interactions(importance_df, feature_names, X_transformed):
    """Generate pairwise interactions between significant features."""
    print("  Generating pairwise interactions between significant features...")
    
    # Get significant features
    sig_features = importance_df[importance_df['Significant']]['Feature'].tolist()
    
    if len(sig_features) < 2:
        print("  Not enough significant features for interactions")
        return None
    
    print(f"  Creating interactions among {len(sig_features)} significant features")
    
    # Get indices of significant features
    sig_indices = [list(feature_names).index(f) for f in sig_features]
    
    # Create list to store interactions
    interactions = []
    
    # Generate all pairwise interactions
    for idx1, idx2 in combinations(sig_indices, 2):
        feat1 = feature_names[idx1]
        feat2 = feature_names[idx2]
        interactions.append({
            'Feature1': feat1,
            'Feature2': feat2,
            'Feature1_idx': idx1,
            'Feature2_idx': idx2,
            'Interaction_Name': f"{feat1} × {feat2}"
        })
    
    print(f"  Generated {len(interactions)} potential interactions")
    return pd.DataFrame(interactions)

def generate_combined_interactions(results, importance_threshold=0.005, output_dir=None):
    """Generate a combined list of potential interactions between task difficulties and network metrics."""
    print("\nGenerating combined interaction list...")
    
    # Get task difficulties and network metrics
    task_metrics = get_task_difficulties()
    network_metrics = get_network_metrics()
    
    # Initialize sets to collect significant features
    significant_task_metrics = set()
    significant_network_metrics = set()
    
    # Track which features exceeded threshold in which analyses
    feature_sources = {}
    
    # Process all results to find important features
    for dataset_name, dataset_results in results.items():
        for outcome, outcome_results in dataset_results.items():
            importance_df = outcome_results['importance']
            
            # Get features exceeding threshold
            important_features = importance_df[importance_df['Importance'] > importance_threshold]['Feature'].tolist()
            
            for feature in important_features:
                # Check if feature is a task metric
                if any(task_metric in feature for task_metric in task_metrics):
                    significant_task_metrics.add(feature)
                    feature_sources[feature] = feature_sources.get(feature, []) + [f"{dataset_name}_{outcome}"]
                
                # Check if feature is a network metric
                if any(network_metric in feature for network_metric in network_metrics):
                    significant_network_metrics.add(feature)
                    feature_sources[feature] = feature_sources.get(feature, []) + [f"{dataset_name}_{outcome}"]
    
    # Log results
    print(f"Found {len(significant_task_metrics)} important task metrics:")
    for metric in significant_task_metrics:
        print(f"  - {metric} (from {', '.join(feature_sources[metric])})")
    
    print(f"\nFound {len(significant_network_metrics)} important network metrics:")
    for metric in significant_network_metrics:
        print(f"  - {metric} (from {', '.join(feature_sources[metric])})")
    
    # Generate all pairwise interactions
    interactions = []
    for task_metric in significant_task_metrics:
        for network_metric in significant_network_metrics:
            interactions.append({
                'Feature1': task_metric,
                'Feature2': network_metric,
                'Feature1_Type': 'Task Difficulty',
                'Feature2_Type': 'Network Metric',
                'Feature1_Sources': ', '.join(feature_sources[task_metric]),
                'Feature2_Sources': ', '.join(feature_sources[network_metric]),
                'Interaction_Name': f"{task_metric} × {network_metric}"
            })
    
    # Convert to DataFrame
    interactions_df = pd.DataFrame(interactions)
    
    # Save combined interactions
    if output_dir:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_file = output_dir / "combined_important_interactions.csv"
        interactions_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(interactions)} potential interactions to {output_file}")
    
    return interactions_df

def save_random_forest_results(outcome, dataset_name, importance_df, interactions_df, output_dir):
    """Save random forest analysis results."""
    # Create output directory
    result_dir = output_dir / f"{dataset_name}_{outcome}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving results to {result_dir}...")
    
    # Save permutation importance with confidence intervals
    importance_df.to_csv(result_dir / "permutation_importance.csv", index=False)
    
    # Save potential interactions if available
    if interactions_df is not None:
        interactions_df.to_csv(result_dir / "potential_interactions.csv", index=False)
    
    print("  Results saved successfully")
    
    return result_dir

def plot_importance_with_ci(importance_df, result_dir):
    """Plot top features with confidence intervals."""
    print("  Plotting feature importance with confidence intervals...")
    
    # Get top 15 features or all if fewer
    top_n = min(15, len(importance_df))
    top_features = importance_df.head(top_n)
    
    # Create plot with appropriate aspect ratio
    plt.figure(figsize=(10, 8))
    
    # Define colors
    significant_color = "#71A33F"  # Green
    non_significant_color = "#B3B3B3"  # Gray
    error_bar_color = "#B3B3B3"  # Gray
    
    # Reverse the order of features for plotting (highest at top)
    y_pos = range(top_n-1, -1, -1)
    
    # Prepare data
    importances = top_features['Importance'].values
    lower_ci = top_features['Lower_CI'].values
    upper_ci = top_features['Upper_CI'].values
    
    # Plot error bars (all in gray)
    plt.errorbar(
        importances, 
        y_pos,
        xerr=np.vstack([
            importances - lower_ci,
            upper_ci - importances
        ]),
        fmt='none',  # No central marker
        ecolor=error_bar_color,
        capsize=5,
        zorder=1
    )
    
    # Plot points with conditional coloring
    for i, importance in enumerate(importances):
        color = significant_color if importance >= 0.01 else non_significant_color
        plt.scatter(importance, y_pos[i], color=color, s=50, zorder=2)
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, zorder=0)
    
    # Labels and formatting
    plt.yticks(y_pos, top_features['Feature'].values)
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance with 95% Confidence Intervals')
    plt.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Use log scale for x-axis
    plt.xscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(result_dir / "feature_importance.pdf")
    plt.savefig(result_dir / "feature_importance.png", dpi=300)
    plt.close()

def run_random_forest_analysis(dataset_path, outcome, categorical_features, continuous_features, output_dir):
    """Run random forest feature importance analysis for a single dataset and outcome."""
    dataset_name = Path(dataset_path).stem
    start_time = time.time()
    
    print(f"\nStarting random forest analysis for {dataset_name} - {outcome}")
    
    # Load data
    print("  Loading dataset...")
    df = load_parquet_dataset(dataset_path)
    
    # Filter features to only include those present in the dataset
    available_categorical = [f for f in categorical_features if f in df.columns]
    available_continuous = [f for f in continuous_features if f in df.columns]
    
    print(f"  Using {len(available_categorical)} categorical and {len(available_continuous)} continuous features")
    print(f"  Full dataset has {len(df)} rows")
    
    # Partition data into training and test sets
    train_indices, test_indices = partition_data(df)
    
    # Preprocess data
    print("  Preprocessing data...")
    preprocessor, X_transformed, feature_names = preprocess_random_forest(
        df, available_categorical, available_continuous
    )
    
    # Split data into sets
    X_train = X_transformed[train_indices]
    y_train = df[outcome].iloc[train_indices].values
    X_test = X_transformed[test_indices]
    y_test = df[outcome].iloc[test_indices].values
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Calculate permutation importance with confidence intervals
    importance_df = calculate_permutation_importance_with_ci(model, X_test, y_test, feature_names)
    
    # Generate potential pairwise interactions between significant features
    interactions_df = generate_pairwise_interactions(importance_df, feature_names, X_transformed)
    
    # Save results
    result_dir = save_random_forest_results(outcome, dataset_name, importance_df, interactions_df, output_dir)
    
    # Plot importance with confidence intervals
    plot_importance_with_ci(importance_df, result_dir)
    
    total_time = time.time() - start_time
    print(f"  Analysis completed in {total_time:.1f} seconds")
    
    # Return key results
    return importance_df, interactions_df

def run_all_random_forest_analyses(study_name="aiteams01nm_20250128_223001"):
    """Run random forest feature importance analyses for all datasets and outcomes."""
    overall_start = time.time()
    output_dir = Path("data/results") / study_name / "random_forest"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define datasets and outcomes
    datasets = [
        f"data/preprocessed/{study_name}/dataset1.parquet", 
        f"data/preprocessed/{study_name}/dataset2.parquet"
    ]
    outcomes = get_outcome_metrics()
    
    # Get categorical features
    categorical_features = get_categorical_variables()
    
    # Get already transformed continuous features directly
    # No need to do any transformation, as it's already done in the dataset
    continuous_features = (
        get_control_variables() +  # Already transformed control variables
        get_network_metrics() +    # Already transformed network metrics
        get_task_difficulties()    # Already transformed task difficulties
    )
    
    print(f"Starting Random Forest analyses for study: {study_name}")
    print(f"Will process {len(datasets)} datasets with {len(outcomes)} outcomes each")
    
    # Run analysis for all combinations
    total_tasks = len(datasets) * len(outcomes)
    completed = 0
    results = {}
    
    for dataset_path in datasets:
        dataset_name = Path(dataset_path).stem
        results[dataset_name] = {}
        
        for outcome in outcomes:
            try:                
                importance_df, interactions_df = run_random_forest_analysis(
                    dataset_path,
                    outcome,
                    categorical_features,
                    continuous_features,
                    output_dir
                )
                
                results[dataset_name][outcome] = {
                    'importance': importance_df,
                    'interactions': interactions_df
                }
                
                print(f"Summary for {dataset_name} - {outcome}:")
                sig_features = importance_df[importance_df['Significant']]
                print(f"  Found {len(sig_features)} statistically significant features")
                
                print(f"  Top 5 features by permutation importance:")
                print(importance_df[['Feature', 'Importance']].head(5).to_string(
                    index=False, 
                    formatters={'Importance': '{:.4f}'.format}
                ))
                
                if interactions_df is not None and len(interactions_df) > 0:
                    print(f"  Generated {len(interactions_df)} potential interactions")
                    print(f"  First 3 potential interactions:")
                    for i, row in interactions_df.head(3).iterrows():
                        print(f"    {i+1}. {row['Feature1']} × {row['Feature2']}")
                
                completed += 1
                print(f"\nTask {completed}/{total_tasks} ({completed/total_tasks*100:.0f}% complete)")
                
            except Exception as e:
                print(f"Error analyzing {dataset_name} - {outcome}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    overall_time = time.time() - overall_start
    print(f"\nAll analyses completed in {overall_time:.1f} seconds")
    print(f"Results saved to {output_dir}")
    
    # Generate combined interaction list
    combined_interactions = generate_combined_interactions(results, importance_threshold=0.005, output_dir=output_dir)
    
    return results, combined_interactions

if __name__ == "__main__":
    study_name = "aiteams01nm_20250128_223001"
    results, combined_interactions = run_all_random_forest_analyses(study_name)