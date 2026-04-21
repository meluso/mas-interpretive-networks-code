# figures/util.py
"""
Created on Thu Mar  4 10:11:05 2021

@author: John Meluso
"""

import matplotlib.colors as mcolors
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import os

def filter_by_significance(df, alpha=0.01):
    """Create a filtered version of coefficients where only significant values remain."""
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Find all columns ending with _p (p-value columns)
    p_value_columns = [col for col in df.columns if col.endswith('_p')]
    
    # For each p-value column, set the corresponding coefficient to zero if not significant
    for p_col in p_value_columns:
        # Construct the coefficient column name
        coef_col = p_col.replace('_p', '_coef')
        
        # Only proceed if the coefficient column exists
        if coef_col in df.columns:
            # Create mask for non-significant values
            mask = df[p_col] >= alpha
            filtered_df.loc[mask, coef_col] = 0.0
    
    return filtered_df

def set_fonts(extra_params={}):
    params = {
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'cm',
        'legend.fontsize': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.titlesize': 8
        }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
    
# Define specific color sets for common use cases
# Hand-picked for optimal visual distinction
GREENS_3 = ['#c5e0b4', '#71A33F', '#3c6b1f']  # Light, medium, dark green
GREENS_4 = ['#e2f0d9', '#a9d08e', '#548235', '#2f5c0f']  # Very light to very dark

def get_gray_green_cmap(name="GrayGn", n_colors=256):
    """
    Returns a gray-green diverging colormap that matches the
    intensity progression of matplotlib's PRGn colormap.
    """
    # Original PRGn data tuple
    _PRGn_data = (
        (0.25098039215686274, 0.0, 0.29411764705882354),
        (0.46274509803921571, 0.16470588235294117, 0.51372549019607838),
        (0.6, 0.4392156862745098, 0.6705882352941176),
        (0.76078431372549016, 0.6470588235294118, 0.81176470588235294),
        (0.90588235294117647, 0.83137254901960789, 0.90980392156862744),
        (0.96862745098039216, 0.96862745098039216, 0.96862745098039216),
        (0.85098039215686272, 0.94117647058823528, 0.82745098039215681),
        (0.65098039215686276, 0.85882352941176465, 0.62745098039215685),
        (0.35294117647058826, 0.68235294117647061, 0.38039215686274508),
        (0.10588235294117647, 0.47058823529411764, 0.21568627450980393),
        (0.0, 0.26666666666666666, 0.10588235294117647)
    )

    # Create GrayGn data by replacing purple values with matching luminance grays
    _GrayGn_data = []
    
    # Process the first 5 values (the purple side)
    for i in range(5):
        rgb = _PRGn_data[i]
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        _GrayGn_data.append((luminance, luminance, luminance))

    # Add the white midpoint
    _GrayGn_data.append(_PRGn_data[5])

    # Add the remaining green values unchanged
    for i in range(6, 11):
        _GrayGn_data.append(_PRGn_data[i])

    # Create and return the colormap
    return mcolors.LinearSegmentedColormap.from_list(name, _GrayGn_data, N=n_colors)

# Add this line if you want the colormap available directly when importing util
GrayGn = get_gray_green_cmap()

# Color generation functions for flexible use cases
def get_green_palette(n=3):
    """Return n shades of green from light to dark."""
    return plt.cm.Greens(np.linspace(0.3, 0.8, n))

def get_diverging_palette(n=11):
    """Return a gray-to-green diverging palette with n colors.
    Negative values are gray, positive values are green, center is white."""
    # Create custom colormap
    neg_colors = plt.cm.Greys(np.linspace(0.3, 0.7, n//2))  # Gray for negative
    pos_colors = plt.cm.Greens(np.linspace(0.3, 0.7, n//2))  # Green for positive
    
    # Create array with white in the middle
    if n % 2 == 0:
        # Even number of colors (divide evenly)
        colors = np.vstack((neg_colors[::-1], pos_colors))
    else:
        # Odd number of colors (with white in the middle)
        mid_color = np.array([[1, 1, 1, 1]])  # White
        colors = np.vstack((neg_colors[::-1], mid_color, pos_colors))
    
    return colors
    
def fig_size(frac_width,frac_height,n_cols=1,n_rows=1):
    
    # Set default sizes
    page_width = 8.5
    page_height = 11
    side_margins = 0.5
    tb_margins = 1
    middle_margin = 0.25
    mid_marg_width = middle_margin*(n_cols-1)
    mid_marg_height = middle_margin*(n_rows-1)
    
    # Width logic
    if frac_width == 1:
        width = page_width - side_margins
    else:
        width = (page_width - side_margins - mid_marg_width)*frac_width
        
    # Height logic
    if frac_height == 1:
        height = page_height - tb_margins
    else:
        height = (page_height - tb_margins - mid_marg_height)*frac_height
        
    return (width,height)

def arrow(ax, xyfrom, xyto, text='', fc='#AAAAAA', ec='#AAAAAA'):
    an = ax.annotate(text=text, xy=xyto, xytext=xyfrom, annotation_clip=False,
        arrowprops=dict(arrowstyle='->',fc=fc,ec=ec),
        xycoords='axes fraction')
    return an

def set_border(ax, top=False, bottom=False, left=False, right=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)

def get_formats():
    return ['jpg','pdf','png']

def save_publication_fig(name, dpi=1200, **kwargs):
    save_fig(name, dpi, fig_type='publication', **kwargs)
    
def save_pub_fig(name, **kwargs):
    save_publication_fig(name, **kwargs)
        
def save_presentation_fig(name, dpi=1200, **kwargs):
    save_fig(name, dpi, fig_type='presentation', **kwargs)

def save_fig(name, dpi=1200, fig_type=None, **kwargs):
    for ff in get_formats():
        if fig_type:
            path = f'figures/{fig_type}/{ff}'
        else: 
            path = f'figures/{ff}'
        if not os.path.exists(path):
            os.makedirs(path)
        fname = f'{path}/{name}.{ff}'
        plt.savefig(fname, format=ff, dpi=dpi, **kwargs)

def get_optimizer(study_name):
    """Extract optimizer code from study name."""
    # Extract characters 9-10 from study name (0-indexed positions 9-10)
    return study_name[9:11]