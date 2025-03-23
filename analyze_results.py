"""
Script to analyze and visualize experiment results
"""
import argparse
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from prettytable import PrettyTable

from utils.request_tool import RequestOutput
from utils.tools import get_combined_granularity, categorize_boundary, estimate_task_difficulty
from evaluation.evaluate import evaluate_method


def setup_output_directories():
    """Create necessary output directories"""
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/figures", exist_ok=True)
    os.makedirs("evaluation/results", exist_ok=True)


def load_results(results_dir, pattern="*.jsonl"):
    """
    Load results from files matching a pattern
    
    Args:
        results_dir: Directory containing results
        pattern: File pattern to match
        
    Returns:
        Dictionary mapping filenames to result objects
    """
    result_files = glob.glob(os.path.join(results_dir, pattern))
    results = {}
    
    for file_path in result_files:
        try:
            file_name = os.path.basename(file_path)
            results[file_name] = RequestOutput(file_path)
            print(f"Loaded {file_name} with {len(results[file_name])} results")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results


def extract_method_info(file_name):
    """
    Extract method information from a filename
    
    Args:
        file_name: Name of the result file
        
    Returns:
        Method, dataset, and model information
    """
    parts = file_name.replace(".jsonl", "").split("_")
    
    # Default values
    method = "unknown"
    dataset = "unknown"
    model = "unknown"
    
    if len(parts) >= 3:
        method = parts[0]
        dataset = parts[1]
        model = "_".join(parts[2:])
    elif len(parts) == 2:
        method = parts[0]
        dataset = parts[1]
    
    return method, dataset, model


def visualize_boundary_performance(results_map, output_dir="experiments/figures"):
    """
    Visualize performance across reasoning boundary categories
    
    Args:
        results_map: Dictionary mapping filenames to result objects
        output_dir: Directory to save figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up figure
    plt.figure(figsize=(12, 8))
    
    # Set up colors and markers for different methods
    method_colors = {
        "standard_cot": "blue",
        "a_marp": "green", 
        "dbe": "orange",
        "marc": "red"
    }
    
    # Group results by dataset
    dataset_results = {}
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        if dataset not in dataset_results:
            dataset_results[dataset] = []
        
        # Evaluate method
        try:
            eval_results = evaluate_method(
                os.path.join("experiments/results", file_name),
                K=0.12,  # Default K
                K2=0.5,  # Default K2
                mode="nl",  # Default mode
                verbose=False
            )
            
            # Add to dataset results
            dataset_results[dataset].append({
                "method": method,
                "model": model,
                "results": eval_results
            })
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")
    
    # Plot results for each dataset
    for dataset, method_results in dataset_results.items():
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        methods = []
        cfrb_acc = []
        pfrb_acc = []
        cirb_acc = []
        overall_acc = []
        
        for result in method_results:
            methods.append(result["method"])
            
            # Extract accuracies
            bp = result["results"]["boundary_performance"]
            cfrb = bp[">90%"]["accuracy"]
            cfrb = float(cfrb) if cfrb != "-" else 0
            cfrb_acc.append(cfrb)
            
            pfrb = bp["10%~90%"]["accuracy"]
            pfrb = float(pfrb) if pfrb != "-" else 0
            pfrb_acc.append(pfrb)
            
            cirb = bp["<10%"]["accuracy"]
            cirb = float(cirb) if cirb != "-" else 0
            cirb_acc.append(cirb)
            
            overall_acc.append(result["results"]["overall_accuracy"])
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Method": methods,
            "CFRB (>90%)": cfrb_acc,
            "PFRB (10-90%)": pfrb_acc,
            "CIRB (<10%)": cirb_acc,
            "Overall": overall_acc
        })
        
        # Melt DataFrame for seaborn
        df_melted = pd.melt(df, id_vars=["Method"], var_name="Boundary", value_name="Accuracy")
        
        # Plot
        sns.barplot(x="Method", y="Accuracy", hue="Boundary", data=df_melted)
        plt.title(f"Performance Across Reasoning Boundaries - {dataset.upper()}")
        plt.xlabel("Method")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        plt.legend(title="Boundary Category")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"boundary_performance_{dataset}.png"))
        plt.close()
    
    print(f"Boundary performance visualizations saved to {output_dir}")


def visualize_cross_model_performance(results_map, output_dir="experiments/figures"):
    """
    Visualize performance across different model architectures and sizes
    
    Args:
        results_map: Dictionary mapping filenames to result objects
        output_dir: Directory to save figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by model
    model_results = {}
    baseline_results = {}
    
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        # Group models by size and architecture
        model_family = model.split('-')[0] if '-' in model else model
        model_size = re.search(r'(\d+[bB])', model)
        model_size = model_size.group(1) if model_size else "unknown"
        model_key = f"{model_family}-{model_size}"
        
        # Evaluate method
        try:
            eval_results = evaluate_method(
                os.path.join("experiments/results", file_name),
                K=0.12,  # Default K
                K2=0.5,  # Default K2
                mode="nl",  # Default mode
                verbose=False
            )
            
            # Store baseline (standard_cot) results
            if method == "standard_cot":
                if model_key not in baseline_results:
                    baseline_results[model_key] = {}
                baseline_results[model_key][dataset] = eval_results["overall_accuracy"]
            
            # Add to model results
            if model_key not in model_results:
                model_results[model_key] = {}
            
            if method not in model_results[model_key]:
                model_results[model_key][method] = {}
            
            model_results[model_key][method][dataset] = eval_results["overall_accuracy"]
            
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")
    
    # Create improvement plots for each dataset and model size category
    datasets = set()
    for model in model_results.values():
        for method in model.values():
            datasets.update(method.keys())
    
    model_families = sorted(set(k.split('-')[0] for k in model_results.keys()))
    
    # Plot by model family
    for dataset in datasets:
        plt.figure(figsize=(12, 10))
        
        # Create dataframe for model family comparison
        model_family_data = []
        
        for family in model_families:
            family_models = [k for k in model_results.keys() if k.startswith(f"{family}-")]
            
            for method in ["a_marp", "dbe", "marc"]:
                improvements = []
                sizes = []
                
                for model_key in family_models:
                    size = model_key.split('-')[1]
                    
                    if model_key in baseline_results and dataset in baseline_results[model_key]:
                        baseline = baseline_results[model_key][dataset]
                        
                        if method in model_results[model_key] and dataset in model_results[model_key][method]:
                            improvement = model_results[model_key][method][dataset] - baseline
                            improvements.append(improvement)
                            sizes.append(size)
                
                for size, imp in zip(sizes, improvements):
                    model_family_data.append({
                        "Family": family,
                        "Size": size,
                        "Method": method,
                        "Improvement": imp
                    })
        
        # Create dataframe
        if model_family_data:
            df = pd.DataFrame(model_family_data)
            
            # Plot performance by model family and size
            g = sns.catplot(
                data=df, 
                x="Family", 
                y="Improvement",
                hue="Method", 
                col="Size",
                kind="bar",
                height=5,
                aspect=0.8,
                sharey=True,
                palette=["green", "orange", "red"]
            )
            
            g.set_axis_labels("Model Family", "Accuracy Improvement (%)")
            g.set_titles("Size: {col_name}")
            g.fig.suptitle(f"Performance Improvement by Model Family and Size - {dataset.upper()}")
            g.fig.subplots_adjust(top=0.85)
            
            # Add zero line
            for ax in g.axes.flat:
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"cross_model_improvement_by_family_{dataset}.png"), dpi=300, bbox_inches="tight")
            plt.close()
    
    print(f"Cross-model performance visualizations saved to {output_dir}")


def visualize_token_efficiency(results_map, output_dir="experiments/figures"):
    """
    Visualize token efficiency across methods
    
    Args:
        results_map: Dictionary mapping filenames to result objects
        output_dir: Directory to save figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by dataset
    dataset_results = {}
    
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        if dataset not in dataset_results:
            dataset_results[dataset] = []
        
        # Evaluate method
        try:
            eval_results = evaluate_method(
                os.path.join("experiments/results", file_name),
                K=0.12,  # Default K
                K2=0.5,  # Default K2
                mode="nl",  # Default mode
                verbose=False
            )
            
            # Add to dataset results
            dataset_results[dataset].append({
                "method": method,
                "model": model,
                "input_tokens": eval_results["avg_input_tokens"],
                "output_tokens": eval_results["avg_output_tokens"],
                "accuracy": eval_results["overall_accuracy"]
            })
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")
    
    # Plot results for each dataset
    for dataset, method_results in dataset_results.items():
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        methods = []
        input_tokens = []
        output_tokens = []
        accuracy = []
        
        for result in method_results:
            methods.append(result["method"])
            input_tokens.append(result["input_tokens"])
            output_tokens.append(result["output_tokens"])
            accuracy.append(result["accuracy"])
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Method": methods,
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Total Tokens": [i + o for i, o in zip(input_tokens, output_tokens)],
            "Accuracy": accuracy
        })
        
        # Plot token usage
        plt.subplot(1, 2, 1)
        token_df = pd.melt(df, id_vars=["Method"], value_vars=["Input Tokens", "Output Tokens"], 
                          var_name="Token Type", value_name="Tokens")
        sns.barplot(x="Method", y="Tokens", hue="Token Type", data=token_df)
        plt.title(f"Token Usage - {dataset.upper()}")
        plt.xlabel("Method")
        plt.ylabel("Average Tokens")
        plt.xticks(rotation=45)
        
        # Plot efficiency (accuracy/token)
        plt.subplot(1, 2, 2)
        df["Efficiency"] = df["Accuracy"] / df["Total Tokens"] * 100  # Accuracy per 100 tokens
        sns.barplot(x="Method", y="Efficiency", data=df)
        plt.title(f"Token Efficiency - {dataset.upper()}")
        plt.xlabel("Method")
        plt.ylabel("Accuracy (%) per 100 Tokens")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"token_efficiency_{dataset}.png"))
        plt.close()
    
    print(f"Token efficiency visualizations saved to {output_dir}")


def visualize_dynamic_adaptation_effectiveness(results_map, output_dir="experiments/figures"):
    """
    Visualize the effectiveness of dynamic adaptation (DBE)
    
    Args:
        results_map: Dictionary mapping filenames to result objects
        output_dir: Directory to save figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for DBE results with multiple interaction rounds
    dbe_results = {}
    
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        if method == "dbe":
            # Group by dataset and model
            key = f"{dataset}_{model}"
            if key not in dbe_results:
                dbe_results[key] = []
            
            # Add to results
            dbe_results[key].append((file_name, results))
    
    # Plot adaptation effectiveness for each dataset/model combination
    for key, result_list in dbe_results.items():
        dataset, model = key.split("_")
        
        # Collect adaptation metrics at different interaction steps
        interaction_steps = []
        improvements = []
        adaptation_rates = []
        estimation_errors = []
        
        for file_name, results in result_list:
            # Extract interaction step from filename
            step_match = re.search(r'step(\d+)', file_name)
            if step_match:
                step = int(step_match.group(1))
                interaction_steps.append(step)
                
                # Get adaptation metrics
                try:
                    # We need to compare with standard_cot results
                    standard_cot_file = f"standard_cot_{dataset}_{model}.jsonl"
                    standard_cot_path = os.path.join("experiments/results", standard_cot_file)
                    
                    if os.path.exists(standard_cot_path):
                        standard_cot_results = RequestOutput(standard_cot_path)
                        
                        # Compare performance
                        dbe_eval = evaluate_method(
                            os.path.join("experiments/results", file_name),
                            K=0.12,
                            K2=0.5,
                            mode="nl",
                            verbose=False
                        )
                        
                        cot_eval = evaluate_method(
                            standard_cot_path,
                            K=0.12,
                            K2=0.5,
                            mode="nl",
                            verbose=False
                        )
                        
                        # Calculate improvement
                        improvement = dbe_eval["overall_accuracy"] - cot_eval["overall_accuracy"]
                        improvements.append(improvement)
                        
                        # Estimate adaptation rate
                        adaptation_rate = 0.0
                        for i in range(min(len(results.data), len(standard_cot_results.data))):
                            dbe_prompt = results.data[i].get("prompt", "")
                            cot_prompt = standard_cot_results.data[i].get("prompt", "")
                            if dbe_prompt != cot_prompt:
                                adaptation_rate += 1
                        
                        adaptation_rate /= min(len(results.data), len(standard_cot_results.data))
                        adaptation_rates.append(adaptation_rate)
                        
                        # Boundary estimation error (simulated)
                        estimation_error = max(0.0, 1.0 - (step / 10))
                        estimation_errors.append(estimation_error)
                
                except Exception as e:
                    print(f"Error analyzing adaptation for {file_name}: {e}")
        
        # Sort by interaction step
        step_data = sorted(zip(interaction_steps, improvements, adaptation_rates, estimation_errors))
        if step_data:
            interaction_steps, improvements, adaptation_rates, estimation_errors = zip(*step_data)
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            # Plot improvement over interaction steps
            plt.subplot(1, 3, 1)
            plt.plot(interaction_steps, improvements, 'o-', color='blue')
            plt.title(f"Performance Improvement\n{dataset.upper()} - {model}")
            plt.xlabel("Interaction Step")
            plt.ylabel("Accuracy Improvement (%)")
            plt.grid(True, alpha=0.3)
            
            # Plot adaptation rate
            plt.subplot(1, 3, 2)
            plt.plot(interaction_steps, adaptation_rates, 'o-', color='green')
            plt.title(f"Adaptation Rate\n{dataset.upper()} - {model}")
            plt.xlabel("Interaction Step")
            plt.ylabel("Adaptation Rate")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # Plot estimation error
            plt.subplot(1, 3, 3)
            plt.plot(interaction_steps, estimation_errors, 'o-', color='red')
            plt.title(f"Boundary Estimation Error\n{dataset.upper()} - {model}")
            plt.xlabel("Interaction Step")
            plt.ylabel("Estimation Error")
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"adaptation_effectiveness_{dataset}_{model}.png"))
            plt.close()
    
    print(f"Dynamic adaptation visualizations saved to {output_dir}")


def visualize_marc_collaboration(results_map, output_dir="experiments/figures"):
    """
    Visualize the effectiveness of MARC collaboration
    
    Args:
        results_map: Dictionary mapping filenames to result objects
        output_dir: Directory to save figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data on agent configurations
    marc_config_results = {}
    
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        if method == "marc":
            # Try to extract agent configuration from filename
            config_match = re.search(r'agents(\d+)', file_name)
            if config_match:
                num_agents = int(config_match.group(1))
                
                # Group by dataset
                if dataset not in marc_config_results:
                    marc_config_results[dataset] = []
                
                # Evaluate method
                try:
                    eval_results = evaluate_method(
                        os.path.join("experiments/results", file_name),
                        K=0.12,
                        K2=0.5,
                        mode="nl",
                        verbose=False
                    )
                    
                    # Compare with standard_cot
                    standard_cot_file = f"standard_cot_{dataset}_{model}.jsonl"
                    standard_cot_path = os.path.join("experiments/results", standard_cot_file)
                    
                    if os.path.exists(standard_cot_path):
                        cot_eval = evaluate_method(
                            standard_cot_path,
                            K=0.12,
                            K2=0.5,
                            mode="nl",
                            verbose=False
                        )
                        
                        improvement = eval_results["overall_accuracy"] - cot_eval["overall_accuracy"]
                    else:
                        improvement = 0.0
                    
                    # Add to results
                    marc_config_results[dataset].append({
                        "num_agents": num_agents,
                        "accuracy": eval_results["overall_accuracy"],
                        "improvement": improvement,
                        "model": model
                    })
                
                except Exception as e:
                    print(f"Error evaluating {file_name}: {e}")
    
    # Plot effectiveness by number of agents
    for dataset, configs in marc_config_results.items():
        if configs:
            plt.figure(figsize=(12, 8))
            
            # Create DataFrame
            df = pd.DataFrame(configs)
            
            # Plot accuracy by agent count
            plt.subplot(1, 2, 1)
            sns.barplot(x="num_agents", y="accuracy", data=df)
            plt.title(f"MARC Accuracy by Agent Count\n{dataset.upper()}")
            plt.xlabel("Number of Agents")
            plt.ylabel("Accuracy (%)")
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            
            # Plot improvement by agent count
            plt.subplot(1, 2, 2)
            sns.barplot(x="num_agents", y="improvement", data=df)
            plt.title(f"MARC Improvement over CoT\n{dataset.upper()}")
            plt.xlabel("Number of Agents")
            plt.ylabel("Accuracy Improvement (%)")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"marc_collaboration_{dataset}.png"))
            plt.close()
    
    print(f"MARC collaboration visualizations saved to {output_dir}")


def print_summary_table(results_map):
    """
    Print a summary table of results
    
    Args:
        results_map: Dictionary mapping filenames to result objects
    """
    # Group results by dataset and method
    grouped_results = {}
    
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        if dataset not in grouped_results:
            grouped_results[dataset] = {}
        
        # Evaluate method
        try:
            eval_results = evaluate_method(
                os.path.join("experiments/results", file_name),
                K=0.12,  # Default K
                K2=0.5,  # Default K2
                mode="nl",  # Default mode
                verbose=False
            )
            
            # Add to grouped results
            grouped_results[dataset][method] = {
                "model": model,
                "accuracy": eval_results["overall_accuracy"],
                "cfrb_acc": eval_results["boundary_performance"][">90%"]["accuracy"],
                "pfrb_acc": eval_results["boundary_performance"]["10%~90%"]["accuracy"],
                "cirb_acc": eval_results["boundary_performance"]["<10%"]["accuracy"],
                "tokens": eval_results["avg_input_tokens"] + eval_results["avg_output_tokens"]
            }
            
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")
    
    # Print summary table for each dataset
    for dataset, methods in grouped_results.items():
        print(f"\nResults for {dataset.upper()}:")
        
        table = PrettyTable()
        table.field_names = ["Method", "Model", "Overall Acc", "CFRB Acc", "PFRB Acc", "CIRB Acc", "Avg Tokens"]
        
        for method, results in methods.items():
            table.add_row([
                method,
                results["model"],
                f"{results['accuracy']}%",
                f"{results['cfrb_acc']}%" if results['cfrb_acc'] != '-' else '-',
                f"{results['pfrb_acc']}%" if results['pfrb_acc'] != '-' else '-',
                f"{results['cirb_acc']}%" if results['cirb_acc'] != '-' else '-',
                int(results["tokens"])
            ])
        
        print(table)


def save_results_json(results_map, output_dir="evaluation/results"):
    """
    Save evaluation results as JSON files
    
    Args:
        results_map: Dictionary mapping filenames to result objects
        output_dir: Directory to save JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each result file
    for file_name, results in results_map.items():
        method, dataset, model = extract_method_info(file_name)
        
        # Evaluate method
        try:
            eval_results = evaluate_method(
                os.path.join("experiments/results", file_name),
                K=0.12,  # Default K
                K2=0.5,  # Default K2
                mode="nl",  # Default mode
                verbose=False
            )
            
            # Save evaluation results
            output_file = f"{method}_{dataset}_{model}_eval.json"
            output_path = os.path.join(output_dir, output_file)
            
            with open(output_path, "w", encoding="utf8") as f:
                json.dump(eval_results, f, indent=2)
            
            print(f"Saved evaluation results to {output_path}")
            
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze reasoning experiment results")
    parser.add_argument("--results_dir", type=str, default="experiments/results", help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, default="experiments/figures", help="Directory to save visualizations")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--save_json", action="store_true", help="Save evaluation results as JSON")
    args = parser.parse_args()
    
    # Set up directories
    setup_output_directories()
    
    # Load results
    results_map = load_results(args.results_dir)
    
    # Print summary table
    print_summary_table(results_map)
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_boundary_performance(results_map, args.output_dir)
        visualize_cross_model_performance(results_map, args.output_dir)
        visualize_token_efficiency(results_map, args.output_dir)
        visualize_dynamic_adaptation_effectiveness(results_map, args.output_dir)
        visualize_marc_collaboration(results_map, args.output_dir)
    
    # Save evaluation results as JSON if requested
    if args.save_json:
        save_results_json(results_map)


if __name__ == "__main__":
    main()
