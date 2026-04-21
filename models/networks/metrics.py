# models/networks/metrics.py
"""
Created on Wed Oct 23, 2024

@author: John Meluso
"""

# Import libraries
import networkx.algorithms.assortativity as assort
import networkx.algorithms.centrality as cent
import networkx.algorithms.cluster as clust
import networkx.algorithms.shortest_paths.generic as paths
import networkx.algorithms.distance_measures as dist
from networkx import classes, NetworkXError
from statistics import mean, stdev
import numpy as np
import warnings

class NetworkMetrics:
    """Handles calculation of network metrics with optimized error handling."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def compute_all(self):
        """Compute all metrics at once to minimize overhead."""
        
        # Calculate vector of centralities
        centrality_degree = self._safe_compute(
            lambda g: list(cent.degree_centrality(g).values())
        )
        centrality_betweenness = self._safe_compute(
            lambda g: list(cent.betweenness_centrality(g).values())
        )
        centrality_eigenvector = self._safe_compute(
            lambda g: list(
                cent.eigenvector_centrality(
                    g, 
                    tol=1e-3, 
                    max_iter=1000
                ).values()
            )
        )
        
        # Calculate vector of nearest neighbor degrees
        nnd = self._safe_compute(
            lambda g: self._safe_neighbor_degrees(g)
        )
        
        # Build metrics dictionary
        metrics = {
            'team_graph_centrality_degree_mean': (self._safe_mean(centrality_degree)),
            'team_graph_centrality_degree_stdev': (self._safe_stdev(centrality_degree)),
            'team_graph_centrality_betweenness_mean': (self._safe_mean(centrality_betweenness)),
            'team_graph_centrality_betweenness_stdev': (self._safe_stdev(centrality_betweenness)),
            'team_graph_centrality_eigenvector_mean': (self._safe_mean(centrality_eigenvector)),
            'team_graph_centrality_eigenvector_stdev': (self._safe_stdev(centrality_eigenvector)),
            'team_graph_nearest_neighbor_degree_mean': (self._safe_mean(nnd)),
            'team_graph_nearest_neighbor_degree_stdev': (self._safe_stdev(nnd)),
            'team_graph_clustering': self._safe_compute(clust.average_clustering),
            'team_graph_density': self._safe_compute(lambda g: classes.function.density(g)),
            'team_graph_assortativity': self._safe_compute(assort.degree_assortativity_coefficient),
            'team_graph_pathlength': self._safe_compute(paths.average_shortest_path_length),
            'team_graph_diameter': self._safe_compute(dist.diameter)
        }
        
        return metrics
    
    def _safe_neighbor_degrees(self, g):
        """Safely compute normalized nearest neighbor degrees."""
        try:
            n = len(g)
            if n <= 1:
                return []
            neighbor_degrees = assort.average_neighbor_degree(g)
            return [knn/(n - 1) if n > 1 else 0 
                   for knn in neighbor_degrees.values()]
        except:
            return []
    
    def _safe_mean(self, values):
        """Safely compute mean of a sequence."""
        try:
            return float(mean(values))
        except:
            return np.nan
    
    def _safe_stdev(self, values):
        """Safely compute standard deviation of a sequence."""
        try:
            return float(stdev(values))
        except:
            return np.nan
    
    def _safe_compute(
        self, 
        metric_func, 
        default_val=np.nan, 
        zero_val=0
    ):
        """Optimized computation with error handling for all common NetworkX
        issues.
        
        Args:
            metric_func: NetworkX metric function to compute
            default_val: Value to return on most errors (default: nan)
            zero_val: Value to return for zero-division cases (default: 0)
            
        Returns:
            Computed metric value or appropriate default/error value
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                return metric_func(self.graph)
            except NetworkXError:
                # Handle disconnected graphs, empty graphs, etc.
                return default_val
            except ValueError:
                # Handle invalid inputs, undefined metrics
                return default_val
            except RuntimeWarning as rw:
                if 'invalid value encountered in' in str(rw):
                    return zero_val
                return default_val
            except:
                return default_val
