"""
correlation_analysis_helper_functions.py

Helper functions for analyzing directional gene-gene correlations and regulatory relationships
in simulated gene regulatory networks (GRNs), particularly based on twin-based inference.

This module includes utilities for:
- Extracting metadata from filenames
- Loading gene interaction matrices and simulation parameters
- Constructing correlation matrices from gene expression simulations
- Visualizing gene-gene correlations as heatmaps or directional network graphs
- Annotating relationships as regulatory or non-regulatory
- Printing categorized summaries for interpretability

Functions
---------
extract_param_index(filename: str) -> str
    Extracts the parameter index string (e.g., '0_1') from a simulation file path.

read_input_matrix(path_to_matrix: str) -> Tuple[int, np.ndarray]
    Loads a gene interaction matrix from a file and returns its shape and contents.

get_param_data(param_df: pd.DataFrame, param_index: str) -> Dict[str, float]
    Retrieves a flat dictionary of kinetic and interaction parameters for a given simulation.

dict_to_matrix(correlation_dict: Dict[str, float], gene_list: List[str]) -> pd.DataFrame
    Converts a flat dictionary of gene-gene correlations to a square matrix DataFrame.

plot_matrix_as_heatmap(corr_matrix: pd.DataFrame, gene_list: List[str], ...)
    Plots a correlation matrix as a heatmap, highlighting regulated and unregulated gene pairs.

print_summary(no_regulation: List[Tuple[str, str]], 
              single_state_regulation: List[Tuple[str, str]], 
              multiple_states_no_reg: List[Tuple[str, str]], 
              multiple_states_and_reg: List[Tuple[str, str]])
    Prints a categorized summary of gene pair relationships.

plot_network(correlation_matrix: pd.DataFrame, gene_list: List[str], title: Optional[str] = None)
    Visualizes gene-gene correlations as a directional network graph, using arrows or flat-headed lines
    to indicate inferred directionality or undetermined regulation.

Helper Functions
----------------
make_reds_blues_colormap() -> matplotlib.colors.Colormap
    Creates a custom red-blue colormap for correlation values.

shrink_arrow_endpoints(...) -> Tuple[Tuple[float, float], Tuple[float, float]]
    Computes arrow start and end coordinates that are offset from node centers.

flat_t_head_arrow(...) -> None
    Draws a repression-like arrow with a flat T-head.

polygon_layout(gene_list: List[str], radius: float = 1.0) -> Dict[str, Tuple[float, float]]
    Assigns circular coordinates to genes for network layout.
"""

#Import packages
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import networkx as nx
import seaborn as sns

def extract_param_index(filename: str) -> str:
    """
    Extracts the parameter row index (e.g., '0_1') from a simulation filename.

    Handles both 'df_row_' and 'df_rows_' prefixes. The extraction stops before
    an 8-digit date stamp (ddmmyyyy) if present.

    Args:
        filename (str): The simulation filename.

    Returns:
        str: The row identifier (e.g., '0_1'), or 'unknown' if the pattern is not found.
    """
    try:
        # Match df_row_ or df_rows_
        match = re.search(r"df_rows?_(\d+(?:_\d+)*)", filename)
        if not match:
            return "unknown"

        core = match.group(1)
        # Remove trailing date if mistakenly included
        parts = core.split("_")
        cleaned = []
        for part in parts:
            if part.isdigit() and len(part) == 8:  # ddmmyyyy date
                break
            cleaned.append(part)

        return "_".join(cleaned) if cleaned else "unknown"
    except Exception:
        return "unknown"

def get_param_data(param_df, param_index, n_genes):
    """
    Extracts and flattens parameters for a given simulation from a parameter DataFrame.

    This function:
      1. Selects specific rows from `param_df` based on `param_index`.
      2. Flattens gene-specific parameters so keys are labeled as `<term>_gene_<id>`.
      3. Calculates degradation rates (`k_deg_mRNA_gene_X`, `k_deg_protein_gene_X`)
         from the corresponding half-life values.
      4. Adds interaction parameters (non-gene-specific) from the first selected row.

    Args:
        param_df (pd.DataFrame):
            Parameter DataFrame containing both gene-level and interaction-level parameters.
            Must contain columns for each gene term in `gene_terms` and optionally extra metadata.
        param_index (str):
            Underscore-separated string of row indices in `param_df` to use.
            Example: "12_13" will select rows 12 and 13.
        n_genes (int):
            Number of genes expected in the simulation; must match the number of selected rows.

    Returns:
        dict:
            A flat dictionary mapping parameter names to values, e.g.:
            {
                "k_on_gene_1": ...,
                "k_off_gene_1": ...,
                "mrna_half_life_gene_1": ...,
                "k_deg_mRNA_gene_1": ...,
                ...
                "<interaction_param>": ...
            }

    Raises:
        AssertionError:
            If the number of selected rows does not match `n_genes`.
    """
    gene_terms = ["k_on", "k_off", "mrna_half_life", "protein_half_life", 
                  "k_prod_protein", "k_prod_mRNA"]
    extra_terms = ["pair_id", "gene_id"]

    rows = [int(i) for i in param_index.split("_")]
    assert len(rows) == n_genes, "Mismatch in number of input genes and parameter rows"

    selected_rows = param_df.iloc[rows]
    param_dict = {}

    # Gene-specific parameters
    for gene_id, row in enumerate(selected_rows.itertuples(index=False), start=1):
        for term in gene_terms:
            key = f"{term}_gene_{gene_id}"
            param_dict[key] = getattr(row, term)

        # Add degradation terms from half_life
        param_dict[f"k_deg_mRNA_gene_{gene_id}"] = np.log(2) / param_dict[f"mrna_half_life_gene_{gene_id}"]
        param_dict[f"k_deg_protein_gene_{gene_id}"] = np.log(2) / param_dict[f"protein_half_life_gene_{gene_id}"]

    # Interaction parameters (from first row)
    interaction_cols = [col for col in param_df.columns if col not in gene_terms + extra_terms]
    interaction_values = param_df.loc[rows[0], interaction_cols]
    for col in interaction_cols:
        param_dict[col] = interaction_values[col]

    return param_dict

def read_input_matrix(path_to_matrix: str) -> (int, np.ndarray):
    """
    Reads an input matrix from a specified file path and returns its dimensions and content.

    Args:
        path_to_matrix (str): The file path to the matrix file. The file should contain
                              a comma-separated matrix of integers.

    Returns:
        tuple: A tuple containing:
            - int: The number of rows in the matrix.
            - np.ndarray: The matrix as a NumPy array. If the matrix is a single value,
                          it is reshaped into a 1x1 array.

    Raises:
        ValueError: If the file cannot be loaded.
    """
    try:
        matrix = np.loadtxt(path_to_matrix, dtype=int, delimiter=',')
        if matrix.ndim == 0:
            matrix = matrix.reshape((1,1))
        return matrix.shape[0], matrix
    except Exception as e:
        raise ValueError(f"Error loading matrix from {path_to_matrix}: {e}")

def dict_to_matrix(correlation_dict, gene_list):
    
    matrix = pd.DataFrame(index=gene_list, columns=gene_list, dtype=float)
    for key, value in correlation_dict.items():
        g1, g2 = key.split("-")
        matrix.loc[g1, g2] = value
    return matrix


def make_reds_blues_colormap():
    reds = plt.cm.Reds(np.linspace(1, 0, 128))   # deep red → white
    blues = plt.cm.Blues(np.linspace(0, 1, 128)) # white → deep blue
    colors = np.vstack((reds, blues))
    return LinearSegmentedColormap.from_list('RedsBlues', colors)



def plot_matrix_as_heatmap(corr_matrix, gene_list, no_regulation=None, potential_regulation=None, title=None, add_gene_labels=True,
                            add_time=False, time=None, gray_out_no_reg=False, vmin=None, vmax=None, cmap=None):
    """
    Plot a gene-gene correlation matrix as a heatmap with regulatory overlays and dynamic formatting.

    This function visualizes a square correlation matrix (e.g., gene-gene correlations at a given timepoint),
    optionally marking entries with no inferred regulation and highlighting potential regulatory pairs.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        A square DataFrame representing gene-gene correlations (e.g., Spearman or directional). 
        Must have both row and column labels matching `gene_list`.

    gene_list : list of str
        List of gene names in the expected order (e.g., ["gene_1", "gene_2", ...]).

    no_regulation : list of tuple, optional
        List of (gene_i, gene_j) pairs where no regulation is inferred. 
        If `gray_out_no_reg` is True, these entries will be grayed out (masked as NaN in the heatmap).

    potential_regulation : list of tuple, optional
        List of (gene_i, gene_j) pairs with possible regulation. These will be marked with solid black rectangles.

    title : str, optional
        Plot title. If `add_time=True`, timepoint annotations will be appended to the title.

    add_gene_labels : bool, default=True
        Whether to annotate the axes with gene labels (e.g., g1, g2, ...).

    add_time : bool, default=False
        Whether to append time information to gene labels on the axes (e.g., $g1_{t1}$).

    time : list or tuple of int, optional
        Time information to annotate gene labels:
        - If `len(time) == 1`, both axes are assumed to correspond to the same time.
        - If `len(time) == 2`, the row and column axes will be labeled with t1 and t2, respectively.
        Required if `add_time=True`.

    gray_out_no_reg : bool, default=False
        If True, entries in `no_regulation` will be masked and shown as gray (NaN in the heatmap).

    vmin : float, optional
        Lower bound of colormap. If not provided, inferred from data.

    vmax : float, optional
        Upper bound of colormap. If not provided, inferred from data.

    cmap : str or matplotlib colormap, optional
        Colormap to use. If not provided:
        - Uses 'Blues' if all values ≥ 0
        - Uses 'Reds' if all values ≤ 0
        - Uses a custom Red-Blue diverging map if values span both negative and positive

    Returns
    -------
    None
        The function displays a heatmap using `matplotlib.pyplot` and does not return any value.

    Notes
    -----
    - The function assumes `corr_matrix` and `gene_list` are consistent (i.e., same order and names).
    - Entries in `no_regulation` that are not in `gene_list` will be silently ignored.
    - The function supports both symmetric and asymmetric matrices (e.g., directional correlations).
    - If `vmin == vmax`, a small epsilon will be added to avoid a degenerate colormap.
    """

    if add_time:
        if time is None or not isinstance(time, (list, tuple)) or len(time) == 0:
            raise ValueError("If add_time=True, you must provide a non-empty list of 1 or 2 time values in `time`.")
        if len(time) > 2:
            raise ValueError("Time can have at most two entries.")

    # Format gene names: gene_1 → g1
    base_names = [g.replace("gene_", "g") for g in gene_list]

    # Format axis labels
    if add_gene_labels:
        if add_time:
            if len(time) == 1:
                row_labels = [f"$g{i}_{{{time[0]}}}$" for i in range(1, len(base_names) + 1)]
                col_labels = row_labels
            else:
                row_labels = [f"$g{i}_{{{time[0]}}}$" for i in range(1, len(base_names) + 1)]
                col_labels = [f"$g{i}_{{{time[1]}}}$" for i in range(1, len(base_names) + 1)]
        else:
            row_labels = base_names
            col_labels = base_names
    else:
        row_labels = [""] * len(gene_list)
        col_labels = [""] * len(gene_list)

    # Prepare plot matrix
    plot_matrix = corr_matrix.copy()

    # --- Handle masking ---
    mask = np.zeros_like(plot_matrix.values, dtype=bool)
    if gray_out_no_reg and no_regulation:
        for g1, g2 in no_regulation:
            if g1 in gene_list and g2 in gene_list:
                i = gene_list.index(g1)
                j = gene_list.index(g2)
                plot_matrix.iloc[i, j] = np.nan
                mask[i, j] = True

    # --- Handle vmin/vmax auto-scaling ---
    data_values = plot_matrix.values.flatten()
    data_values = data_values[~np.isnan(data_values)]

    if len(data_values) == 0:
        vmin, vmax = -1.0, 1.0
    else:
        if vmin is None:
            vmin = np.min(data_values)
        if vmax is None:
            vmax = np.max(data_values)
        if vmin == vmax:
            vmin -= 1e-4
            vmax += 1e-4

    # --- Choose colormap adaptively ---
    from matplotlib.colors import TwoSlopeNorm

    if cmap is None and vmin < 0 and vmax > 0:
        cmap = make_reds_blues_colormap()
        center_span = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-center_span, vcenter=0.0, vmax=center_span)

    else:
        norm = None
        if cmap is None:
            cmap = "Blues" if vmin >= 0 else "Reds"

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,  # ✅ Will be None if not diverging
        xticklabels=col_labels,
        yticklabels=row_labels,
        square=True,
        cbar_kws={'label': 'Correlation'},
        linewidths=0.5,
        linecolor='lightgray',
        mask=mask
    )
    if norm is not None:
        cbar = heatmap.collections[0].colorbar
        # cbar.set_clim(vmin, vmax)  # display only actual data range on colorbar


    # --- Add regulation boxes ---
    if potential_regulation:
        for g1, g2 in potential_regulation:
            if g1 in gene_list and g2 in gene_list:
                i = gene_list.index(g1)
                j = gene_list.index(g2)
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(rect)

    # --- Title ---
    if title:
        if add_time:
            if len(time) == 1:
                title += f" @ time {time[0]}h"
            elif len(time) == 2:
                title += f" (rows: t{time[0]}, cols: t{time[1]})"
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.show()


def print_summary(no_regulation, 
                  single_state_regulation, 
                  multiple_states_no_reg, 
                  multiple_states_and_reg):
    """
    Prints a structured summary of gene pair classifications.

    Parameters
    ----------
    no_regulation : list of tuple
        Gene pairs with no inferred regulation.

    single_state_regulation : list of tuple
        Gene pairs with single-state regulation.

    multiple_states_no_reg : list of tuple
        Gene pairs with multiple states but no additional regulation evidence.

    multiple_states_and_reg : list of tuple
        Gene pairs with multiple states and additional evidence of regulation.

    Returns
    -------
    None
    """
    def print_section(title, pairs):
        print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
        if not pairs:
            print("  (none)")
        else:
            for g1, g2 in pairs:
                print(f"  {g1} - {g2}")

    print_section("1. No Regulation", no_regulation)
    print_section("2. Single-State Regulation", single_state_regulation)
    print_section("3. Multiple States (No Regulation)", multiple_states_no_reg)
    print_section("4. Multiple States with Regulation", multiple_states_and_reg)

# --- Helpers ---
def make_reds_blues_colormap():
    reds = plt.cm.Reds(np.linspace(1, 0, 128))   # deep red → white
    blues = plt.cm.Blues(np.linspace(0, 1, 128)) # white → deep blue
    colors = np.vstack((reds, blues))
    return LinearSegmentedColormap.from_list('RedsBlues', colors)

def shrink_arrow_endpoints(x1, y1, x2, y2, shrink_source=0.2, shrink_target=0.2, lateral_offset=0.0, scaled_shrink_val = 0.25):
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)

    if dist == 0:
        return (x1, y1), (x2, y2)

    scaled_shrink = scaled_shrink_val * dist  # 25% of distance, adjust as needed
    shrink_source = scaled_shrink
    shrink_target = scaled_shrink*0.8

    ux, uy = dx / dist, dy / dist
    orth_x, orth_y = -uy, ux
    x1_off = x1 + shrink_source * ux + lateral_offset * orth_x
    y1_off = y1 + shrink_source * uy + lateral_offset * orth_y
    x2_off = x2 - shrink_target * ux + lateral_offset * orth_x
    y2_off = y2 - shrink_target * uy + lateral_offset * orth_y
    return (x1_off, y1_off), (x2_off, y2_off)

def flat_t_head_arrow(start, end, color='red', linewidth=2, rad=0.2, abs_weight=0.5, ax=None):
    """Draw repression arrow with T-head scaled by abs_weight."""
    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle='-',
        color=color,
        linewidth=linewidth,
        zorder=1
    )
    ax.add_patch(arrow)

    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    if dist == 0:
        return

    ux, uy = dx / dist, dy / dist
    tx, ty = x2, y2
    px, py = -uy, ux

    # Scale T-head size with abs_weight (capped)
    t_len = 0.03 + 0.07 * min(abs_weight, 1.0)
    t_xs = [tx - px * t_len, tx + px * t_len]
    t_ys = [ty - py * t_len, ty + py * t_len]
    ax.plot(t_xs, t_ys, color=color, linewidth=linewidth, solid_capstyle='round', zorder=2)

def polygon_layout(gene_list, radius=1.0):
    n = len(gene_list)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return {
        gene_list[i]: (radius * np.cos(angle), radius * np.sin(angle))
        for i, angle in enumerate(angles)
    }

def plot_network(correlation_matrix, gene_list, title=None):
    DG = nx.DiGraph()
    for gene in gene_list:
        DG.add_node(gene)

    # Add all gene-gene interactions (even if directionless)
    # Add all gene-gene interactions (only if both exist in correlation_matrix)
    for g1 in gene_list:
        for g2 in gene_list:
            if g1 == g2:
                continue
            if g1 in correlation_matrix.index and g2 in correlation_matrix.columns:
                val = correlation_matrix.loc[g1, g2]
                if pd.notna(val):
                    DG.add_edge(g1, g2, weight=val)


    # Define node positions and colors
    pos = polygon_layout(gene_list, radius=max(2, len(gene_list) / 2)) if len(gene_list) > 2 else {
        gene_list[0]: (0, 0),
        gene_list[1]: (1, 0)
    }

    fig, ax = plt.subplots(figsize=(9, 9))
    node_colors = np.array([
    correlation_matrix.loc[g, g] if g in correlation_matrix.index and g in correlation_matrix.columns else 0
    for g in gene_list
    ])

    v = 1
    norm = Normalize(vmin=-v, vmax=v)
    cmap = make_reds_blues_colormap()
    node_rgba = cmap(norm(node_colors))
    node_rgba[:, -1] = 0.8

    nx.draw_networkx_nodes(DG, pos, ax=ax, node_color=node_rgba, node_size=6000, edgecolors='black', linewidths=1.5)

    labels = {node: f"$g{int(node.split('_')[-1])}$" for node in DG.nodes()}
    nx.draw_networkx_labels(DG, pos, labels=labels, font_size=16, ax=ax)

    # Draw edges
    for u, v in DG.edges():
        raw_weight = correlation_matrix.loc[u, v]
        if pd.isnull(raw_weight) or raw_weight == 0:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        offset = -0.15 if DG.has_edge(v, u) else 0.0
        rad = 0.2 if DG.has_edge(v, u) else 0.0
        start, end = shrink_arrow_endpoints(x1, y1, x2, y2, lateral_offset=offset)
        color = cmap(norm(raw_weight))
        abs_weight = abs(raw_weight)

        if raw_weight > 0:
            head_length = min(10 + abs_weight * 10, 20)
            head_width = min(10 + abs_weight * 10, 20)
            arrow_style = f'->,head_length={head_length},head_width={head_width}'

            # Positive correlation → activation (arrow)
            arrow = FancyArrowPatch(
                start, end,
                connectionstyle=f'arc3,rad={rad}',
                arrowstyle=arrow_style,
                mutation_scale=1,
                color=color,
                linewidth=min(abs_weight * 10, 5.0),
                zorder=1
            )
            ax.add_patch(arrow)
        elif raw_weight < 0:
            # Negative correlation → repression (flat T-bar)
            flat_t_head_arrow(
                start, end,
                color=color,
                linewidth=min(abs_weight * 10, 5.0),
                rad=rad,
                abs_weight=abs_weight,
                ax=ax
            )


    ax.set_title(title or "Inferred GRN", fontsize=16, fontweight='bold')
    ax.axis('off')

    if len(pos) > 0:
        x_vals, y_vals = zip(*pos.values())
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        ax.set_xlim(min(x_vals) - 0.3 * x_range, max(x_vals) + 0.3 * x_range)
        ax.set_ylim(min(y_vals) - 0.3 * y_range, max(y_vals) + 0.3 * y_range)

    if len(gene_list) == 2:
        ax.set_xlim(-0.5, len(gene_list) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    else:
        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

    plt.tight_layout()
    plt.show()




