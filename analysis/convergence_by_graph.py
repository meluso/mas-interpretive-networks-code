#!/usr/bin/env python3
# analysis/convergence_by_graph.py
"""Analyze convergence timing patterns by graph type and rationality bounds.

Investigates whether complete graph performance patterns relate to convergence
timing, calculating mean steps to convergence across graph types and bounds.
"""

import pandas as pd
from pathlib import Path


def analyze_convergence_timing(study_name, team_size=8):
    """Calculate convergence statistics by graph type and rationality bound.
    
    Args:
        study_name: Name of study to analyze
        team_size: Team size to filter (default 8)
        
    Returns:
        Tuple of (full data, overall stats, pivot table)
    """
    # Load mean differences data
    data_path = Path("data/results") / study_name / "mean_differences.csv"
    df = pd.read_csv(data_path)
    
    # Filter to specified team size
    df = df[df['team_size'] == team_size].copy()
    df['agent_steplim_rounded'] = df['agent_steplim'].round(3)
    
    print("="*80)
    print("CONVERGENCE TIMING ANALYSIS")
    print("="*80)
    
    # Overall convergence by graph type
    print("\n1. MEAN CONVERGENCE STEPS BY GRAPH TYPE (all bounds)")
    print("-"*80)
    overall_conv = df.groupby('graph_slug')['convergence_step_mean'].mean().sort_values()
    print(overall_conv.to_string())
    
    complete_mean = overall_conv.loc['complete']
    dataset_mean = overall_conv.mean()
    complete_rank = (overall_conv <= complete_mean).sum()
    
    print(f"\nComplete graph mean: {complete_mean:.2f} steps")
    print(f"Dataset mean: {dataset_mean:.2f} steps")
    print(f"Complete rank: {complete_rank} of {len(overall_conv)}")
    
    # Convergence by rationality bound
    print("\n\n2. CONVERGENCE STEPS BY RATIONALITY BOUND")
    print("-"*80)
    
    bounds = sorted(df['agent_steplim_rounded'].unique())
    
    for bound in bounds:
        print(f"\nBound = {bound}")
        print("-"*40)
        
        bound_data = df[df['agent_steplim_rounded'] == bound]
        bound_conv = bound_data.sort_values('convergence_step_mean')[
            ['graph_slug', 'convergence_step_mean']
        ]
        print(bound_conv.to_string(index=False))
        
        # Complete graph comparison
        complete_row = bound_data[bound_data['graph_slug'] == 'complete']
        if len(complete_row) > 0:
            complete_val = complete_row['convergence_step_mean'].values[0]
            bound_mean = bound_data['convergence_step_mean'].mean()
            rank = (bound_data['convergence_step_mean'] <= complete_val).sum()
            
            print(f"\n  Complete: {complete_val:.2f} (rank {rank}/{len(bound_data)})")
            print(f"  Mean: {bound_mean:.2f}")
            print(f"  Δ: {complete_val - bound_mean:+.2f}")
    
    # Pivot table
    print("\n\n3. PIVOT TABLE: Graph × Rationality Bound")
    print("-"*80)
    
    pivot = df.pivot_table(
        values='convergence_step_mean',
        index='graph_slug',
        columns='agent_steplim_rounded',
        aggfunc='mean'
    )
    pivot['mean_all'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean_all')
    
    print(pivot.round(2).to_string())
    
    # Early convergence test
    print("\n\n4. EARLY CONVERGENCE HYPOTHESIS")
    print("-"*80)
    print("\nDoes complete graph converge earlier than average?\n")
    
    for bound in bounds:
        bound_data = df[df['agent_steplim_rounded'] == bound]
        complete_row = bound_data[bound_data['graph_slug'] == 'complete']
        
        if len(complete_row) > 0:
            complete_conv = complete_row['convergence_step_mean'].values[0]
            mean_conv = bound_data['convergence_step_mean'].mean()
            pct_diff = ((complete_conv / mean_conv) - 1) * 100
            converges_early = "YES" if complete_conv < mean_conv else "NO"
            
            print(f"Bound {bound:.3f}:")
            print(f"  Complete: {complete_conv:.2f}")
            print(f"  Average: {mean_conv:.2f}")
            print(f"  Δ: {pct_diff:+.1f}%")
            print(f"  Early: {converges_early}\n")
    
    # Summary statistics for citation
    print("\n5. STATISTICS FOR CITATION")
    print("-"*80)
    
    complete_data = df[df['graph_slug'] == 'complete']
    others_data = df[df['graph_slug'] != 'complete']
    
    complete_overall = complete_data['convergence_step_mean'].mean()
    others_overall = others_data['convergence_step_mean'].mean()
    
    print(f"\nComplete graph: {complete_overall:.2f} steps")
    print(f"Other graphs: {others_overall:.2f} steps")
    print(f"Δ: {complete_overall - others_overall:+.2f} ({((complete_overall/others_overall)-1)*100:+.1f}%)")
    
    print("\nBy Rationality Bound:")
    for bound in bounds:
        bound_data = df[df['agent_steplim_rounded'] == bound]
        complete_subset = bound_data[bound_data['graph_slug'] == 'complete']
        others_subset = bound_data[bound_data['graph_slug'] != 'complete']
        
        if len(complete_subset) > 0 and len(others_subset) > 0:
            comp_mean = complete_subset['convergence_step_mean'].values[0]
            other_mean = others_subset['convergence_step_mean'].mean()
            diff_pct = ((comp_mean / other_mean) - 1) * 100
            
            print(f"  {bound:.3f}: Complete={comp_mean:.1f}, Others={other_mean:.1f}, Δ={diff_pct:+.1f}%")
    
    return df, overall_conv, pivot


if __name__ == '__main__':
    study_name = 'aiteams01nm_20250128_223001'
    df, overall_stats, pivot = analyze_convergence_timing(study_name)
