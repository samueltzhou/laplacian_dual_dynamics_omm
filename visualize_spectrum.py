# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import gymnasium as gym
import src.env
from src.env.wrapper.norm_obs import NormObs
from src.env.grid.utils import load_eig

# import jax # Not used in this script directly

if __name__ == "__main__":
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
    all_eig = []
    
    # Ensure the output directory for spectrum plots exists
    output_dir_spectrum = "./eig_spectrum"
    os.makedirs(output_dir_spectrum, exist_ok=True)

    for env_name in ENV_NAMES:
        # path_txt_grid = f"./src/env/grid/txts/{env_name}.txt" # No longer needed here
        path_eig = f"./src/env/grid/eigval/{env_name}.npz"

        eig_data, eig_not_found = load_eig(path_eig)
        if eig_not_found:
            print(f"Eigenvalues not found for {env_name}, skipping spectrum plot.")
            continue
        
        eigval, eigvec = eig_data
        all_eig.append(eig_data) 

        # Plotting the spectrum for the current environment
        plt.figure(figsize=(10, 6))
        ranks = np.arange(1, len(eigval) + 1)
        
        num_top_eigenvalues = min(11, len(eigval))
        
        plt.scatter(ranks[:num_top_eigenvalues], eigval[:num_top_eigenvalues], color='blue', s=20, label=f"Top {num_top_eigenvalues} Eigenvalues")
        
        if len(eigval) > num_top_eigenvalues:
            plt.scatter(ranks[num_top_eigenvalues:], eigval[num_top_eigenvalues:], color='red', s=10, label="Other Eigenvalues")
        
        plt.title(f"Eigenvalue Spectrum for {env_name}")
        plt.xlabel("Eigenvalue Rank (Largest to Smallest)")
        plt.ylabel("Eigenvalue")
        plt.grid(True)
        plt.legend(loc='upper left')

        largest_eig_text = f"Largest: {eigval[0]:.4f}"
        if len(eigval) >= 11:
            eleventh_eig_text = f"11th Largest: {eigval[10]:.4f}"
        else:
            eleventh_eig_text = f"11th Largest: N/A (<11 total)"
        smallest_eig_text = f"Smallest: {eigval[-1]:.4f}"
        
        annotation_text = f"{largest_eig_text}\n{eleventh_eig_text}\n{smallest_eig_text}"
        
        plt.text(0.98, 0.98, annotation_text,
                 transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plot_filename_spectrum = os.path.join(output_dir_spectrum, f"{env_name.replace('/', '_')}_spectrum.png")
        try:
            plt.savefig(plot_filename_spectrum)
            print(f"Saved spectrum plot for {env_name} to {plot_filename_spectrum}")
        except Exception as e:
            print(f"Error saving spectrum plot for {env_name}: {e}")
        plt.close() 
        
    print("\nFinished generating spectrum plots.")

    
    
