import json
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ttest_ind_from_stats
from tabulate import tabulate

ENV_NAMES_UPPERCASE = [
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

ENV_NAMES_LOWERCASE = [
    "gridmaze_7",
    "gridmaze_9",
    "gridmaze_17",
    "gridmaze_19",
    "gridmaze_26",
    "gridmaze_32",
    "gridroom_1",
    "gridroom_4",
    "gridroom_16",
    "gridroom_32",
    "gridroom_64",
    "gridroomsym_4",
]

# results/visuals/GridRoomSym-4/gridroomsym_4_allo_5_12_25_run_1

def compute_stats(run_type: str, num_runs: int = 10):
    """
    run_type should look like "allo_5_12_25" or something of the sort
    """
    
    all_stats = {
        "run_type": run_type,
        "num_runs": num_runs
    }

    for env_idx in range(len(ENV_NAMES_LOWERCASE)):
        env_name_lowercase = ENV_NAMES_LOWERCASE[env_idx]
        env_name_uppercase = ENV_NAMES_UPPERCASE[env_idx]
        
        env_cos_similarities = []
        print(f"Processing environment: {env_name_uppercase}")

        for run_idx in range(1, num_runs + 1):
            json_path = f"./results/visuals/{env_name_uppercase}/{env_name_lowercase}_{run_type}_run_{run_idx}/training_history.json"
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                final_cos_sim = data.get("final_cos_sim")
                if final_cos_sim is not None:
                    env_cos_similarities.append(final_cos_sim)
                else:
                    print(f"  Warning: 'final_cos_sim' not found in {json_path}")
            except FileNotFoundError:
                print(f"  Warning: File not found {json_path}")
            except json.JSONDecodeError:
                print(f"  Warning: Could not decode JSON from {json_path}")
            except Exception as e:
                print(f"  Warning: An unexpected error occurred with {json_path}: {e}")

        if env_cos_similarities:
            mean_sim = np.mean(env_cos_similarities)
            std_dev_sim = np.std(env_cos_similarities)
            max_sim = np.max(env_cos_similarities)
            
            all_stats[env_name_lowercase] = {
                "mean_cos_sim": mean_sim,
                "std_dev_cos_sim": std_dev_sim,
                "max_cos_sim": max_sim
            }
            print(f"  Stats for {env_name_lowercase}: Mean={mean_sim:.4f}, StdDev={std_dev_sim:.4f}, Max={max_sim:.4f}")
        else:
            print(f"  No data found for {env_name_lowercase} to compute statistics.")
            all_stats[env_name_lowercase] = {
                "mean_cos_sim": None,
                "std_dev_cos_sim": None,
                "max_cos_sim": None
            }
            
    # Save the results
    output_dir = "./results/summary/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{run_type}_summary.json")
    
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
        
    print(f"\nSummary statistics saved to {output_path}")
           
def comparison_bar_chart(run_type_list: list[str]):
    """
    Generates a bar chart comparing mean cosine similarities for different run_types.
    Each run_type in the list should have a corresponding summary JSON file.
    """
    if not run_type_list:
        print("Warning: run_type_list is empty. No chart will be generated.")
        return

    all_run_data = {}
    env_names = ENV_NAMES_LOWERCASE # Assuming all run_types cover these environments

    for run_type in run_type_list:
        summary_file_path = f"./results/summary/{run_type}_summary.json"
        try:
            with open(summary_file_path, 'r') as f:
                data = json.load(f)
            all_run_data[run_type] = {
                "mean_sims": {env: data.get(env, {}).get("mean_cos_sim") for env in env_names},
                "std_devs": {env: data.get(env, {}).get("std_dev_cos_sim") for env in env_names},
                "num_runs": data.get("num_runs", "N/A")
            }
        except FileNotFoundError:
            print(f"Warning: Summary file not found for run_type '{run_type}' at {summary_file_path}. Skipping this run_type.")
            continue
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {summary_file_path} for run_type '{run_type}'. Skipping this run_type.")
            continue

    if not all_run_data:
        print("No data loaded for any run_type. Cannot generate chart.")
        return

    n_groups = len(env_names)
    n_bars = len(all_run_data)
    fig_width = max(10, n_groups * n_bars * 0.5) # Adjust width based on number of bars
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    index = np.arange(n_groups)
    bar_width = 0.8 / n_bars  # Adjust bar width based on number of run_types

    colors = plt.cm.get_cmap('viridis', n_bars + 2) 

    for i, (run_type, data) in enumerate(all_run_data.items()):
        means = [data["mean_sims"].get(env, 0) or 0 for env in env_names] 
        stds = [data["std_devs"].get(env, 0) or 0 for env in env_names] # Get std_devs, default to 0 if missing
        label_text = f"{run_type} (N={data['num_runs']} runs)"
        ax.bar(index + i * bar_width, means, bar_width, yerr=stds, label=label_text, color=colors(i / n_bars), capsize=3)

    ax.set_xlabel('Environment', fontweight='bold')
    ax.set_ylabel('Mean Cosine Similarity', fontweight='bold')
    ax.set_title('Comparison of Mean Cosine Similarities by Run Type', fontweight='bold')
    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(env_names, rotation=45, ha="right")
    ax.legend(title="Run Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05) 

    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    output_dir = "./results/summary/"
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%m%d")
    chart_filename = f"bar_chart_{current_date}.png"
    chart_path = os.path.join(output_dir, chart_filename)
    
    try:
        plt.savefig(chart_path, bbox_inches='tight')
        print(f"\nBar chart saved to {chart_path}")
    except Exception as e:
        print(f"Error saving bar chart: {e}")
    plt.close(fig) 
    
def comparison_table(base_run_type: str, hypothesis_run_type: str, significance_level: float = 0.05):
    """
    Generates a table comparing mean cosine similarities between base_run_type and hypothesis_run_type.
    Uses their standard deviations to perform an independent t-test to determine if 
    the hypothesis_run_type is statistically significantly better than the base_run_type.
    """
    summary_base_path = f"./results/summary/{base_run_type}_summary.json"
    summary_hypo_path = f"./results/summary/{hypothesis_run_type}_summary.json"

    try:
        with open(summary_base_path, 'r') as f:
            data_base = json.load(f)
    except FileNotFoundError:
        print(f"Error: Summary file for base run type '{base_run_type}' not found at {summary_base_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {summary_base_path}")
        return

    try:
        with open(summary_hypo_path, 'r') as f:
            data_hypo = json.load(f)
    except FileNotFoundError:
        print(f"Error: Summary file for hypothesis run type '{hypothesis_run_type}' not found at {summary_hypo_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {summary_hypo_path}")
        return

    table_data = []
    headers = ["Env", f"{base_run_type}", f"{hypothesis_run_type}", "t-statistic", "p-value"]

    num_runs_base_overall = data_base.get("num_runs")
    num_runs_hypo_overall = data_hypo.get("num_runs")

    print(f"Comparing Base: {base_run_type} (Overall N={num_runs_base_overall}) vs. Hypothesis: {hypothesis_run_type} (Overall N={num_runs_hypo_overall})")
    if num_runs_base_overall != num_runs_hypo_overall:
        # Ensure we have numbers to compare, otherwise default to a placeholder for n_base/n_hypo if one is None
        n_base_comp = num_runs_base_overall if num_runs_base_overall is not None else -1
        n_hypo_comp = num_runs_hypo_overall if num_runs_hypo_overall is not None else -1
        if n_base_comp != n_hypo_comp:
             print("Warning: The overall 'num_runs' in the summary files differ. Tests will use these differing overall numbers.")

    for i, env_key_lower in enumerate(ENV_NAMES_LOWERCASE):
        env_key_upper = ENV_NAMES_UPPERCASE[i]
        
        env_data_base = data_base.get(env_key_lower, {})
        env_data_hypo = data_hypo.get(env_key_lower, {})

        mean_base = env_data_base.get("mean_cos_sim")
        std_base = env_data_base.get("std_dev_cos_sim")
        n_base = num_runs_base_overall 

        mean_hypo = env_data_hypo.get("mean_cos_sim")
        std_hypo = env_data_hypo.get("std_dev_cos_sim")
        n_hypo = num_runs_hypo_overall

        row = [env_key_upper]

        if mean_base is not None and std_base is not None:
            row.append(f"{mean_base:.4f} ({std_base:.4f})")
        else:
            row.append("N/A")

        hypo_val_str = "N/A"
        t_stat_str = "N/A"
        p_val_str = "N/A"

        # Ensure all necessary values for t-test are not None
        if all(v is not None for v in [mean_hypo, std_hypo, n_hypo, mean_base, std_base, n_base]):
            t_stat, p_val_two_sided = ttest_ind_from_stats(
                mean1=mean_hypo, std1=std_hypo, nobs1=n_hypo,
                mean2=mean_base, std2=std_base, nobs2=n_base,
                equal_var=False
            )

            if t_stat > 0:
                p_one_sided = p_val_two_sided / 2
            else:
                p_one_sided = 1 - (p_val_two_sided / 2)
            
            t_stat_str = f"{t_stat:.4f}"
            p_val_str = f"{p_one_sided:.4f}"

            hypo_val_formatted = f"{mean_hypo:.4f} ({std_hypo:.4f})"
            if mean_hypo > mean_base and p_one_sided < significance_level:
                hypo_val_str = f"\033[1m{hypo_val_formatted}\033[0m"
            else:
                hypo_val_str = hypo_val_formatted
        else:
            # This else corresponds to the `if all(...)` for t-test values
            if mean_hypo is not None and std_hypo is not None: # hypo data is present but base might be missing for t-test
                 hypo_val_str = f"{mean_hypo:.4f} ({std_hypo:.4f})"
            # t_stat_str and p_val_str remain "N/A"
        
        row.append(hypo_val_str)
        row.append(t_stat_str)
        row.append(p_val_str)
        table_data.append(row)

    table_data_file = []
    for row_console in table_data:
        row_file = [col.replace('\033[1m', '').replace('\033[0m', '') if isinstance(col, str) else col for col in row_console]
        table_data_file.append(row_file)

    table_string_console = tabulate(table_data, headers=headers, tablefmt="heavy_outline")
    table_string_file = tabulate(table_data_file, headers=headers, tablefmt="simple")

    print("\nComparison Table:")
    print(table_string_console)

    output_dir = "./results/summary/"
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%m%d")
    table_filename = f"comparison_table_{base_run_type}_vs_{hypothesis_run_type}_{current_date}.txt"
    table_path = os.path.join(output_dir, table_filename)

    try:
        with open(table_path, 'w') as f:
            f.write(f"Comparison: Base Run Type: {base_run_type} vs. Hypothesis Run Type: {hypothesis_run_type}\n")
            f.write(f"Significance Level for highlighting (console): {significance_level}\n\n")
            f.write(table_string_file)
        print(f"\nTable saved to {table_path}")
    except Exception as e:
        print(f"Error saving table: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute summary statistics, generate comparison charts, and create comparison tables for experiment runs.")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for compute_stats
    stats_parser = subparsers.add_parser('compute', help='Compute and save summary statistics for a single run_type.')
    stats_parser.add_argument("run_type", type=str, help="The type of run to process (e.g., 'allo_5_10_25').")
    stats_parser.add_argument("--num_runs", type=int, default=10, help="Number of runs per environment.")

    # Subparser for comparison_bar_chart
    chart_parser = subparsers.add_parser('chart', help='Generate a comparison bar chart from multiple run_type summary files.')
    chart_parser.add_argument("run_types", nargs='+', type=str, help="A list of run_types to compare (e.g., 'type1 type2 type3').")

    # Subparser for comparison_table
    table_parser = subparsers.add_parser('table', help='Generate a comparison table between two run_types.')
    table_parser.add_argument("base_run_type", type=str, help="The base run_type for comparison.")
    table_parser.add_argument("hypothesis_run_type", type=str, help="The hypothesis run_type to compare against the base.")
    table_parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for the t-test (default: 0.05).")
    
    args = parser.parse_args()
    
    if args.command == 'compute':
        compute_stats(args.run_type, args.num_runs)
    elif args.command == 'chart':
        comparison_bar_chart(args.run_types)
    elif args.command == 'table':
        comparison_table(args.base_run_type, args.hypothesis_run_type, args.alpha)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
    
# python compute_stats.py table allo_5_12_25 effseq_diagpenalty_5_12_25
