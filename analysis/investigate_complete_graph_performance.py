# analysis/investigate_complete_graph_performance.py
"""Investigate why complete graph underperforms on coordinate despite optimal properties.

Compares network properties of graphs that outperform complete to understand
which structural differences explain the performance gap.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config.metrics import NETWORK_METRICS

def load_network_properties_by_graph(study_name, team_size=8):
    """Load aggregated network properties for each graph type at specified team size."""
    file_path = Path(f"data/results/{study_name}/network_properties/network_metrics_by_graph.parquet")
    properties = pd.read_parquet(file_path)
    return properties[properties['team_size'] == team_size].copy()

def load_graph_regression_results(study_name, dataset='dataset2', outcome='convergence_performance'):
    """Load graph-specific OLS regression results with interactions."""
    file_path = Path(f"data/results/{study_name}/ols_regressions/graph_slug_regressions.csv")
    results = pd.read_csv(file_path)
    
    # Filter to interaction models only (main + task + interaction effects)
    results = results[
        (results['dataset'] == dataset) & 
        (results['outcome'] == outcome) &
        (results['model_type'] == 'with_interactions')
    ].copy()
    
    return results

def calculate_conditional_effects(regression_results, task_difficulty):
    """Calculate task conditional effects (main + interaction) for each graph type."""
    conditional_effects = []
    
    for _, row in regression_results.iterrows():
        graph_type = row['main_effect_name']
        
        # Extract coefficients: conditional = main + interaction (task main is in intercept)
        main_coef = row.get('main_effect_coef', 0)
        interaction_coef = row.get(f'interaction_{task_difficulty}_coef', 0)
        
        conditional_effects.append({
            'graph_slug': graph_type,
            'main_effect': main_coef,
            'interaction_effect': interaction_coef,
            'conditional_effect': main_coef + interaction_coef
        })
    
    return pd.DataFrame(conditional_effects)

def compare_properties_to_reference(properties_df, reference_graph='complete', comparison_graphs=None):
    """Calculate property differences between comparison graphs and reference graph."""
    reference_row = properties_df[properties_df['graph_slug'] == reference_graph].iloc[0]
    
    # Build list of metric columns to compare (all mean values)
    metric_columns = [f'{metric}_mean' for metric in NETWORK_METRICS.keys()]
    
    comparisons = []
    
    # Default to comparing all graphs if none specified
    if comparison_graphs is None:
        comparison_graphs = properties_df['graph_slug'].unique()
    
    for graph in comparison_graphs:
        if graph == reference_graph:
            continue
            
        comparison_row = properties_df[properties_df['graph_slug'] == graph].iloc[0]
        
        # Calculate differences for each metric
        differences = {'graph_slug': graph}
        for metric_col in metric_columns:
            reference_value = reference_row[metric_col]
            comparison_value = comparison_row[metric_col]
            
            # Handle NaN values (e.g., assortativity for complete graph)
            if pd.isna(reference_value) or pd.isna(comparison_value):
                difference = np.nan
            else:
                difference = comparison_value - reference_value
            
            differences[f'{metric_col}_diff'] = difference
            differences[f'{metric_col}_ref'] = reference_value
            differences[f'{metric_col}_comp'] = comparison_value
        
        comparisons.append(differences)
    
    return pd.DataFrame(comparisons)

def main():
    """Investigate complete graph performance puzzle through property comparison."""
    study_name = "aiteams01nm_20250128_223001"
    team_size = 8
    effect_threshold = 0.25  # Threshold for meaningful performance differences
    
    print("="*80)
    print("Complete Graph Performance Investigation")
    print("="*80)
    
    # Load network properties and regression results
    print("\n1. Loading network properties and regression results...")
    properties = load_network_properties_by_graph(study_name, team_size)
    regression_results = load_graph_regression_results(study_name)
    
    # Calculate coordinate task conditional effects
    print("\n2. Calculating coordinate task conditional effects...")
    coord_task = 'norm_team_fn_diff_interdep'
    coordinate_effects = calculate_conditional_effects(regression_results, coord_task)
    coordinate_effects = coordinate_effects.sort_values('conditional_effect', ascending=False)
    
    print("\nCoordinate Task Conditional Effects (relative to complete = 0):")
    print(coordinate_effects[['graph_slug', 'conditional_effect']].to_string(index=False))
    
    # Group graphs by performance relative to complete
    print(f"\n3. Grouping graphs by performance (threshold = ±{effect_threshold})...")
    outperformers = coordinate_effects[
        coordinate_effects['conditional_effect'] > effect_threshold
    ]['graph_slug'].tolist()
    similar_performers = coordinate_effects[
        (coordinate_effects['conditional_effect'] >= -effect_threshold) & 
        (coordinate_effects['conditional_effect'] <= effect_threshold)
    ]['graph_slug'].tolist()
    underperformers = coordinate_effects[
        coordinate_effects['conditional_effect'] < -effect_threshold
    ]['graph_slug'].tolist()
    
    print(f"\nOutperformers (>{effect_threshold}): {outperformers}")
    print(f"Similar (±{effect_threshold}): {similar_performers}")
    print(f"Underperformers (<-{effect_threshold}): {underperformers}")
    
    # Compare outperformers' properties to complete
    print("\n4. Comparing outperformer properties to complete graph...")
    property_comparison = compare_properties_to_reference(properties, 'complete', outperformers)
    
    # Create readable summary table
    print("\n--- KEY PROPERTY DIFFERENCES (Outperformer - Complete) ---")
    print("(Positive values = outperformer has MORE of this property)")
    
    summary_table = pd.DataFrame({
        'Graph': property_comparison['graph_slug'],
        'Density': property_comparison['team_graph_density_mean_diff'].round(3),
        'Decentral': property_comparison['team_graph_centrality_eigenvector_mean_mean_diff'].round(3),
        'PathLength': property_comparison['team_graph_pathlength_mean_diff'].round(3),
        'Betw.Mean': property_comparison['team_graph_centrality_betweenness_mean_mean_diff'].round(3),
        'Betw.StDev': property_comparison['team_graph_centrality_betweenness_stdev_mean_diff'].round(3),
        'Deg.StDev': property_comparison['team_graph_centrality_degree_stdev_mean_diff'].round(3),
        'Effect': coordinate_effects.set_index('graph_slug').loc[
            property_comparison['graph_slug']
        ]['conditional_effect'].round(3).values
    })
    
    # Sort by performance effect
    summary_table = summary_table.sort_values('Effect', ascending=False)
    
    # Show complete graph's actual property values
    print("\n5. Complete graph's baseline property values:")
    complete_properties = properties[properties['graph_slug'] == 'complete'].iloc[0]
    
    baseline_values = pd.DataFrame({
        'Property': ['Density', 'Decentralization', 'Path Length', 'Betweenness Mean', 
                     'Betweenness StDev', 'Degree StDev', 'Eigenvector StDev'],
        'Complete Value': [
            complete_properties['team_graph_density_mean'],
            complete_properties['team_graph_centrality_eigenvector_mean_mean'],
            complete_properties['team_graph_pathlength_mean'],
            complete_properties['team_graph_centrality_betweenness_mean_mean'],
            complete_properties['team_graph_centrality_betweenness_stdev_mean'],
            complete_properties['team_graph_centrality_degree_stdev_mean'],
            complete_properties['team_graph_centrality_eigenvector_stdev_mean']
        ]
    })
    baseline_values['Complete Value'] = baseline_values['Complete Value'].round(4)
    
    print("\n" + baseline_values.to_string(index=False))
    print("\nNote: Complete has ZERO variation properties (all nodes identical)")
    
    # Calculate negotiate effects for context
    print("\n6. Negotiate task conditional effects (for contrast with coordinate)...")
    negotiate_task = 'norm_team_fn_diff_alignment'
    negotiate_effects = calculate_conditional_effects(regression_results, negotiate_task)
    negotiate_effects = negotiate_effects.sort_values('conditional_effect', ascending=False)
    
    print("\nTop 10 performers on NEGOTIATE (all perform worse than complete = 0):")
    negotiate_top10 = negotiate_effects[['graph_slug', 'conditional_effect']].head(10).copy()
    negotiate_top10['conditional_effect'] = negotiate_top10['conditional_effect'].round(3)
    print(negotiate_top10.to_string(index=False))
    
    print("\nNote: ALL graphs perform worse than complete on negotiate (all negative)")
    print("      This contrasts with coordinate where most outperform complete")
    
    # Save detailed results
    output_dir = Path(f"data/results/{study_name}/complete_graph_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    coordinate_effects.to_csv(output_dir / "coordinate_conditional_effects.csv", index=False)
    property_comparison.to_csv(output_dir / "property_comparison.csv", index=False)
    negotiate_effects.to_csv(output_dir / "negotiate_conditional_effects.csv", index=False)
    
    print(f"\n7. Detailed results saved to {output_dir}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()