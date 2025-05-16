import numpy as np
import matplotlib.pyplot as plt
import os

ENV_NAMES = [
    "GridMaze-7",
    "GridMaze-9",
    "GridMaze-17",
    "GridMaze-19",
    "GridMaze-26",
    "GridMaze-32",
    "GridRoom-1",
    "GridRoom-4",
    "GridRoom-16",
    "GridRoom-32",
    "GridRoom-64",
    "GridRoomSym-4",
]

def load_grid_from_txt(txt_path):
    '''Loads a grid from a text file.
    Assumes 'W' represents a wall and other characters free space.
    Returns a 2D NumPy array (1 for wall, 0 for free space).
    '''
    print(f"--- Loading grid from: {txt_path} ---")
    raw_lines = []
    with open(txt_path, 'r') as f:
        for i, line_content in enumerate(f):
            stripped_line = line_content.strip()
            if i < 5 : # Print first 5 lines for inspection
                print(f"  Raw line {i}: '{stripped_line}'")
            if stripped_line:
                raw_lines.append(list(stripped_line))
    
    if not raw_lines:
        print("  Warning: No non-empty lines found in file.")
        return np.array([], dtype=np.int8) # Return empty array if file was empty or only whitespace

    # Diagnostic: Print unique characters found in the first few raw lines
    if raw_lines:
        sample_chars = set()
        for r_idx, r_line in enumerate(raw_lines):
            if r_idx < 5: # Check first 5 lines
                for char_in_line in r_line:
                    sample_chars.add(char_in_line)
        print(f"  Unique characters in first 5 non-empty lines: {sample_chars}")

    # Ensure consistent parsing: 1 for wall ('X'), 0 for free space (' ').
    grid = np.array([[1 if cell.upper() == 'X' else 0 for cell in row] for row in raw_lines], dtype=np.int8)
    
    print(f"  Parsed grid shape: {grid.shape}")
    print(f"  Unique values in parsed grid: {np.unique(grid)}")
    if grid.size > 0 and np.all(grid == 1):
        print("  DIAGNOSTIC: Parsed grid is ALL 1s (walls). Check wall character assumption ('X').")
    elif grid.size > 0 and np.all(grid == 0):
        print("  DIAGNOSTIC: Parsed grid is ALL 0s (free space). Check free space character assumption.")
    # print(grid) # Uncomment to see the full grid if small enough
    print("--- Finished loading grid ---")
    return grid

def plot_grid_on_ax(ax, env_name, grid_array):
    '''Plots a single grid layout onto a given matplotlib Axes object.'''
    # cmap='binary' maps 0 to white and 1 to black by default.
    # vmin/vmax ensures this mapping is strict for 0 and 1.
    ax.imshow(grid_array, cmap='binary', origin='upper', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(env_name, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == "__main__":
    output_dir_grids = "./results/summary/grids/"
    os.makedirs(output_dir_grids, exist_ok=True)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    axes = axes.flatten()

    print(f"Starting grid layout generation. Output directory: {output_dir_grids}")
    
    for i, env_name in enumerate(ENV_NAMES):
        if i >= len(axes):
            print(f"Warning: More environments than available subplots. Skipping {env_name}.")
            break

        ax = axes[i]
        path_txt_grid = f"./src/env/grid/txts/{env_name}.txt"
        # print(f"Processing: {env_name}") # Reduced verbosity, load_grid_from_txt will print
        try:
            grid_array = load_grid_from_txt(path_txt_grid)
            if grid_array.size == 0:
                print(f"  Warning: Loaded empty grid for {env_name}. Skipping plot.")
                ax.set_title(f"{env_name}\n(Empty Grid)", fontsize=10, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
                continue
        
            plot_grid_on_ax(ax, env_name, grid_array)
        except FileNotFoundError:
            print(f"  Grid TXT file not found for {env_name} at {path_txt_grid}, skipping grid plot.")
            ax.set_title(f"{env_name}\n(File Not Found)", fontsize=10, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as e:
            print(f"  Error processing grid layout for {env_name} {path_txt_grid}: {e}")
            ax.set_title(f"{env_name}\n(Error)", fontsize=10, color='red')
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(len(ENV_NAMES), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=0.5, h_pad=3.0)
    
    combined_plot_filename = os.path.join(output_dir_grids, "all_grids_layout.png")
    try:
        plt.savefig(combined_plot_filename, dpi=150)
        print(f"\nCombined grid layout plot saved to {combined_plot_filename}")
    except Exception as e:
        print(f"Error saving combined grid layout plot: {e}")
    plt.close(fig)

    print("\nFinished generating grid layout plots.")
