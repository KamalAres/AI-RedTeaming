import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns

# Color palette
htb_green = "#9fef00"
node_black = "#141d2b"
hacker_grey = "#a4b1cd"
white = "#ffffff"
azure = "#0086ff"       # Class 0
nugget_yellow = "#ffaf00" # Class 1
malware_red = "#ff3e3e"    # Class 2
vivid_purple = "#9f00ff"   # Highlight/Accent
aquamarine = "#2ee7b6"   # Highlight/Accent

# Configure plot styles
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": node_black,
        "axes.facecolor": node_black,
        "axes.edgecolor": hacker_grey,
        "axes.labelcolor": white,
        "text.color": white,
        "xtick.color": hacker_grey,
        "ytick.color": hacker_grey,
        "grid.color": hacker_grey,
        "grid.alpha": 0.1,
        "legend.facecolor": node_black,
        "legend.edgecolor": hacker_grey,
        "legend.frameon": True,
        "legend.framealpha": 0.8, # Slightly transparent legend background
        "legend.labelcolor": white,
        "figure.figsize": (12, 7), # Default figure size
    }
)

# Seed for reproducibility - MUST BE 1337
SEED = 1337
np.random.seed(SEED)

print("Setup complete. Libraries imported and styles configured.")

# Generate 3-class synthetic data
n_samples = 1500
centers_3class = [(0, 6), (4, 3), (8, 6)]  # Centers for three blobs
X_3c, y_3c = make_blobs(
    n_samples=n_samples,
    centers=centers_3class,
    n_features=2,
    cluster_std=1.15, # Standard deviation of clusters
    random_state=SEED,
)

# Standardize features
scaler = StandardScaler()
X_3c_scaled = scaler.fit_transform(X_3c)

# Split data into training and testing sets, stratifying by class
X_train_3c, X_test_3c, y_train_3c, y_test_3c = train_test_split(
    X_3c_scaled, y_3c, test_size=0.3, random_state=SEED, stratify=y_3c
)

print(f"\nGenerated {n_samples} samples with 3 classes.")
print(f"Training set size: {X_train_3c.shape[0]} samples.")
print(f"Testing set size: {X_test_3c.shape[0]} samples.")
print(f"Classes: {np.unique(y_3c)}")
print(f"Feature shape: {X_train_3c.shape}")
def plot_data_multi(
    X,
    y,
    title="Multi-Class Dataset Visualization",
    highlight_indices=None,
    highlight_markers=None,
    highlight_colors=None,
    highlight_labels=None,
):
    """
    Plots a 2D multi-class dataset with class-specific colors and optional highlighting.
    Automatically ensures points marked with 'P' are plotted above all others.

    Args:
        X (np.ndarray): Feature data (n_samples, 2).
        y (np.ndarray): Labels (n_samples,).
        title (str): The title for the plot.
        highlight_indices (list | np.ndarray, optional): Indices of points in X to highlight. Defaults to None.
        highlight_markers (list, optional): Markers for highlighted points (recycled if shorter).
                                          Points with marker 'P' will be plotted on top. Defaults to ['o'].
        highlight_colors (list, optional): Edge colors for highlighted points (recycled). Defaults to [vivid_purple].
        highlight_labels (list, optional): Labels for highlighted points legend (recycled). Defaults to [''].
    """
    plt.figure(figsize=(12, 7))
    # Define colors based on the global palette for classes 0, 1, 2 (or more if needed)
    class_colors = [
        azure,
        nugget_yellow,
        malware_red,
    ]  # Extend if you have more than 3 classes
    unique_classes = np.unique(y)
    max_class_idx = np.max(unique_classes) if len(unique_classes) > 0 else -1
    if max_class_idx >= len(class_colors):
        print(
            f"{malware_red}Warning:{white} More classes ({max_class_idx + 1}) than defined colors ({len(class_colors)}). Using fallback color."
        )
        class_colors.extend([hacker_grey] * (max_class_idx + 1 - len(class_colors)))

    cmap_multi = plt.cm.colors.ListedColormap(class_colors)

    # Plot all non-highlighted points first
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap_multi,
        edgecolors=node_black,
        s=50,
        alpha=0.7,
        zorder=1,  # Base layer
    )

    # Plot highlighted points on top if specified
    highlight_handles = []
    if highlight_indices is not None and len(highlight_indices) > 0:
        num_highlights = len(highlight_indices)
        # Provide defaults if None
        _highlight_markers = (
            highlight_markers
            if highlight_markers is not None
            else ["o"] * num_highlights
        )
        _highlight_colors = (
            highlight_colors
            if highlight_colors is not None
            else [vivid_purple] * num_highlights
        )
        _highlight_labels = (
            highlight_labels if highlight_labels is not None else [""] * num_highlights
        )

        for i, idx in enumerate(highlight_indices):
            if not (0 <= idx < X.shape[0]):
                print(
                    f"{malware_red}Warning:{white} Invalid highlight index {idx} skipped."
                )
                continue

            # Determine marker, edge color, and label for this point
            marker = _highlight_markers[i % len(_highlight_markers)]
            edge_color = _highlight_colors[i % len(_highlight_colors)]
            label = _highlight_labels[i % len(_highlight_labels)]

            # Determine face color based on the point's true class
            point_class = y[idx]
            try:
                face_color = class_colors[int(point_class)]
            except (IndexError, TypeError):
                print(
                    f"{malware_red}Warning:{white} Class index '{point_class}' invalid. Using fallback."
                )
                face_color = hacker_grey

            current_zorder = (
                3 if marker == "P" else 2
            )  # If marker is 'P', use zorder 3, else 2

            # Plot the highlighted point
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                facecolors=face_color,
                edgecolors=edge_color,
                marker=marker,  # Use the determined marker
                s=180,
                linewidths=2,
                alpha=1.0,
                zorder=current_zorder,  # Use the zorder determined by the marker
            )
            # Create legend handle if label exists
            if label:
                highlight_handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker=marker,
                        color="w",
                        label=label,
                        markerfacecolor=face_color,
                        markeredgecolor=edge_color,
                        markersize=10,
                        linestyle="None",
                        markeredgewidth=1.5,
                    )
                )

    plt.title(title, fontsize=16, color=htb_green)
    plt.xlabel("Feature 1 (Standardized)", fontsize=12)
    plt.ylabel("Feature 2 (Standardized)", fontsize=12)

    # Create class legend handles
    class_handles = []
    unique_classes_present = sorted(np.unique(y))
    for class_idx in unique_classes_present:
        try:
            int_class_idx = int(class_idx)
            class_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Class {int_class_idx}",
                    markersize=10,
                    markerfacecolor=class_colors[int_class_idx],
                    markeredgecolor=node_black,
                    linestyle="None",
                )
            )
        except (IndexError, TypeError):
            print(
                f"{malware_red}Warning:{white} Cannot create legend entry for class {class_idx}."
            )

    # Combine legends
    all_handles = class_handles + highlight_handles
    if all_handles:
        plt.legend(handles=all_handles, title="Classes & Points")

    plt.grid(True, color=hacker_grey, linestyle="--", linewidth=0.5, alpha=0.3)
    plt.show()


# Plot the initial clean training data
print("\n--- Visualizing Clean Training Data ---")
plot_data_multi(X_train_3c, y_train_3c, title="Original Training Data (3 Classes)")
