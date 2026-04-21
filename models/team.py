# models/team.py
"""
Team class implementing a networked team of agents optimizing a shared
objective.

The Team class represents a group of agents connected in a network structure,
where each agent controls a subset of decision variables and makes local
optimization decisions based on neighborhood objectives.

Created on Fri Oct  1 16:42:59 2021
@author: John Meluso
"""

# Import libraries
import networkx as nx
from numpy import abs, array, concatenate, nan
from numpy.random import default_rng
from scipy.stats import spearmanr as corr
from statistics import mean

# Import model files
from models.agent import Agent
import models.objectives.objective as Objective
from models.networks.generators import get_graph
from models.networks.metrics import NetworkMetrics

# Create the random number generator
rng = default_rng()

class Team(nx.Graph):
    """A networked team of agents optimizing a shared objective function."""
    
    def __init__(self, team_size, team_graph_type, agent_num_vars,
                 agent_steplim, agent_optim_type, fn_type,
                 team_graph_opts=None, agent_optim_opts=None, fn_opts=None):
        """Initialize team as a networkx graph with distributed decision
        variables."""
        # Inherit nx.Graph stuff but with additional inputs
        super().__init__()
        
        # Store key parameters
        self.team_size = team_size
        self.agent_num_vars = agent_num_vars
        self.fn_type = fn_type
        self.fn_opts = fn_opts
        
        # Set number of monte carlo trials to run
        self.mc_trials = 100
        
        # Build network from inputs
        self.build_network(team_graph_type, team_size, team_graph_opts)
        
        # Declare team objective
        self.build_team()
        
        # Build agents
        self.build_agents(agent_optim_type, agent_optim_opts, agent_steplim)
        
        # Initialize team state and metrics
        self.initialize_state()
        self.calc_metrics()
    
    def build_network(self, team_graph_type, team_size, team_graph_opts):
        """Build the network structure for the team."""
        graph = get_graph(team_graph_type, team_size, **(team_graph_opts or {}))
        self.add_nodes_from(graph)
        self.add_edges_from(graph.edges)
    
    def build_team(self):
        """Build team objective function from specifications."""
        self.set_team_ks()
        self.set_team_fn()
                
    def build_agents(self, agent_optim_type, agent_optim_opts, agent_steplim):
        """Build Agent instances for each node in the network."""
        for ag in self:
            
            # Create objective function for this agent's neighborhood
            nbhd_fn = get_objective(self.fn_type, self.fn_opts, 
                                    self.agent_num_vars, self.get_nbhd_ks(ag))
            
            # Create agent with its objective function
            self.nodes[ag]['agent'] = Agent(
                num_vars=self.agent_num_vars,
                optimizer_type=agent_optim_type,
                optimizer_opts=agent_optim_opts,
                step_limit=agent_steplim,
                objective_fn=nbhd_fn,
                rng=rng
            )
                
    def initialize_state(self):
        """Initialize team state and agent objectives."""
        
        # Cache current states
        current_states = {
            ag: self.nodes[ag]['agent'].get_x().copy()
            for ag in self
        }
        
        # Initialize neighborhood objectives using cached states
        for ag in self:
            agent = self.nodes[ag]['agent']
            nbhd_state = concatenate(
                [current_states[i] for i in [ag] + list(self.neighbors(ag))])
            agent.set_fx(nbhd_state)
        
        # Initialize team performance and productivity
        self.set_team_fx(self.eval_team_fn(self.get_team_xs()))
        self.set_team_dfdt(nan)
    
    def step(self):
        """Execute one optimization step for the entire team."""
        
        # Get old fx
        fx_old = self.get_team_fx()
        
        # Cache current states and pre-compute neighborhood states
        nbhd_states = self.get_nbhd_states()
        
        # Step 1: Have each agent optimize using their optimizer
        for ag in self:
            agent = self.nodes[ag]['agent']
            
            new_x = agent.step(nbhd_states[ag])
            agent.set_x(new_x)
            
        # Cache current states and pre-compute neighborhood states
        new_nbhd_states = self.get_nbhd_states()
            
        # Step 2: Update neighborhood performances with new states
        for ag in self:
            agent = self.nodes[ag]['agent']
            agent.set_fx(new_nbhd_states[ag])
                    
        # Update system performance and productivity
        fx_curr = self.eval_team_fn(self.get_team_xs())
        self.set_team_fx(fx_curr)
        self.set_team_dfdt((fx_curr-fx_old)/len(self))
    
    
    ### Metric Calculators ###############################################
    
    def calc_metrics(self):
        """Calculate network and function metrics for the team."""
        self.calc_network_metrics()
        self.calc_fn_metrics()
        
    def calc_network_metrics(self):
        """Calculate metrics describing team network structure."""
        network_metrics = NetworkMetrics(self)
        self.network_metrics = network_metrics.compute_all()
    
    def calc_fn_metrics(self):
        """Calculate metrics objective difficulties."""
        
        # Generate random samples
        self.gen_mc_samples()
        
        # Calculate integral difficulty
        self.calc_integral()
        
        # Calculate peaks difficulty
        self.calc_peaks()
        
        # Calculate alignment and interdependence difficulties
        self.calc_align_interdep()
        
    def gen_mc_samples(self):
        """Generate Monte Carlo samples and calculate objective difficulties."""
        
        # Generate and store samples
        self.mc_samples = rng.uniform(
            0, 1, (self.mc_trials, self.team_size*self.agent_num_vars))
        
        # Calculate objective values and integrals for each neighborhood
        for ag in self:
            # Get variable indices for this agent's neighborhood
            agent = self.nodes[ag]['agent']
            nbhd_indices = self.get_nbhd_var_indices(ag)
            
            # Calculate neighborhood objective values using appropriate variable slice
            nbhd_fx = array([
                agent.objective_fn(xx[nbhd_indices]) 
                for xx in self.mc_samples
            ])
            
            # Store results
            self.set_nbhd_mc_fx(ag, nbhd_fx)
            
    def calc_integral(self):
        """Calculate area above the function (1 - mean(team_fx))."""
        team_fx = array([self.eval_team_fn(xx) for xx in self.mc_samples])
        self.set_team_integral(1 - team_fx.mean())
        
    def calc_peaks(self):
        """Get & set number of peaks in team objective from function."""
        self.set_team_peaks(self.team_fn.get_peaks())
    
    def calc_align_interdep(self):
        """Calculate alignment and interdependence between all variable pairs."""
        alignments = []
        interdeps = []
        
        # Calculate for all pairs of agents
        for ii in range(self.team_size):
            for jj in range(ii + 1, self.team_size):
                
                # Get objective values for variables i and j
                fx_ii = self.get_nbhd_mc_fx(ii)
                fx_jj = self.get_nbhd_mc_fx(jj)
                
                # Calculate metrics
                interdep, _ = corr(fx_ii, fx_jj)
                alignment = 1 - mean(abs(fx_ii - fx_jj))
                
                alignments.append(alignment)
                interdeps.append(interdep)
        
        # Store average metrics
        self.set_team_alignment(alignments)
        self.set_team_interdep(interdeps)
    
    
    ### Team Methods ###################################################
    
    def set_team_ks(self):
        '''Set all degrees for the team'''
        self.team_ks = array([kk for nn, kk in self.degree])
    
    def get_team_ks(self):
        '''Get all of the degrees for the team.'''
        return self.team_ks
    
    def get_team_xs(self):
        '''Get all x's for the team concatenated into a single vector.'''
        return concatenate([self.nodes[ag]['agent'].get_x().copy() for ag in self.nodes])
    
    def set_team_fx(self, val):
        '''Set team f(x) with val.'''
        self.team_performance = val
    
    def get_team_fx(self):
        '''Get team f(x) from the team performance attribute.'''
        return self.team_performance
    
    def set_team_dfdt(self, val):
        '''Set team df(x)/dt with val.'''
        self.team_productivity = val
        
    def get_team_dfdt(self):
        '''Get team df(x)/dt from the team productivity attribute.'''
        return self.team_productivity
    
    def set_team_fn(self):
        '''Set team objective from function type and options.'''
        self.team_fn = get_objective(self.fn_type, self.fn_opts,
                                     self.agent_num_vars, self.get_team_ks())
    
    def eval_team_fn(self, xx):
        '''Evaluate the team's objective function.'''
        return self.team_fn(xx)
    
    ### Neighborhood Methods ###########################################
    
    def get_nbhd_var_indices(self, ag):
        """Get indices of variables controlled by agent and its neighbors."""
        
        # Start with agent's own variables
        var_start = sum(self.agent_num_vars for ii in range(ag))
        var_end = var_start + self.agent_num_vars
        indices = list(range(var_start, var_end))
        
        # Add variables controlled by neighbors
        for nbr in self.neighbors(ag):
            nbr_start = sum(self.agent_num_vars for jj in range(nbr))
            nbr_end = nbr_start + self.agent_num_vars
            indices.extend(range(nbr_start, nbr_end))
        
        return indices
    
    def get_nbhd_states(self):
        '''Get the state of each neighborhood, including each agent's x's and
        their neighbors x's.'''
        xs = {ag: self.nodes[ag]['agent'].get_x().copy() for ag in self}
        nbhd_states = {ag: [xs[ag]] + [xs[nbr] for nbr in self.neighbors(ag)]
            for ag in self}
        return nbhd_states
    
    def get_nbhd_ks(self, ag):
        '''Get k's with respect to a specified agent as a numpy array.'''
        k_vect = [self.degree[nbr] for nbr in self.neighbors(ag)]
        k_vect.insert(0, self.degree[ag])
        return array(k_vect)
    
    def get_nbhd_xs_mc(self, ag):
        '''Get x's of an agent and its neighbors.'''
        x_vect = [self.mc_samples[:,nb] for nb in self.neighbors(ag)]
        x_vect.insert(self.ind_ag, self.mc_samples[:,ag])
        return array(x_vect).transpose()
    
    def set_nbhd_mc_fx(self, ag, val):
        """Store Monte Carlo samples of neighborhood objective values."""
        self.nodes[ag]['nbhd_mc_fx'] = val
        
    def get_nbhd_mc_fx(self, ag):
        """Get Monte Carlo samples of neighborhood objective values."""
        return self.nodes[ag]['nbhd_mc_fx']
    
    
    ### Edge Methods ################################################
    
    def set_edge_alignment(self, ag1, ag2, val):
        """Set the alignment value between two agents."""
        self.edges[ag1, ag2]['alignment'] = val
        
    def get_edge_alignment(self, ag1, ag2):
        """Get the alignment value between two agents."""
        return self.edges[ag1, ag2]['alignment']
        
    def set_edge_interdep(self, ag1, ag2, val):
        """Set the interdependence value between two agents."""
        self.edges[ag1, ag2]['interdep'] = val
        
    def get_edge_interdep(self, ag1, ag2):
        """Get the interdependence value between two agents."""
        return self.edges[ag1, ag2]['interdep']
    
    
    ### Function Metric Getters & Setters ############################
    
    def set_team_integral(self, val):
        """Set the team integral difficulty measure."""
        self.team_fn_diff_integral = val
        
    def set_team_peaks(self, val):
        """Set the team peaks difficulty measure."""
        self.team_fn_diff_peaks = val
    
    def set_team_alignment(self, vals):
        """Set the team alignment difficulty measure."""
        try:
            self.team_fn_diff_alignment = mean(vals)
        except:
            if self.team_size == 1:
                self.team_fn_diff_alignment = 1
            else:
                self.team_fn_diff_alignment = nan
        
    def set_team_interdep(self, vals):
        """Set the team interdependence difficulty measure."""
        try:
            self.team_fn_diff_interdep = mean(vals)
        except:
            if self.team_size == 1:
                self.team_fn_diff_interdep = 0
            else:
                self.team_fn_diff_interdep = nan
    
    
def get_objective(fn_type, fn_opts, num_vars, degrees):
    '''Creates and returns a callable object of the specified function type.'''
    try:
        return getattr(Objective, fn_type.capitalize())(fn_opts, num_vars, degrees)
    except:
        raise RuntimeError(f'Function type {fn_type} is not valid.')
