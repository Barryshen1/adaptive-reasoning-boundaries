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
from utils.tools import get_combined_granularity, categorize_boundary
from evaluation.evaluate import evaluate_method


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
                if model not in baseline_results:
                    baseline_results[model] = {}
                baseline_results[model][dataset] = eval_results["overall_accuracy"]
            
            # Add to model results
            if model not in model_results:
                model_results[model] = {}
            
            if method not in model_results[model]:
                model_results[model][method] = {}
            
            model_results[model][method][dataset] = eval_results["overall_accuracy"]
            
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")
    
    # Create improvement plots for each dataset
    datasets = set()
    for model in model_results.values():
        for method in model.values():
            datasets.update(method.keys())
    
    for dataset in datasets:
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        models = []
        a_marp_imp = []
        dbe_imp = []
        marc_imp = []
        
        for model, methods in model_results.items():
            if model in baseline_results and dataset in baseline_results[model]:
                baseline = baseline_results[model][dataset]
                models.append(model)
                
                # Calculate improvements
                if "a_marp" in methods and dataset in methods["a_marp"]:
                    a_marp_imp.append(methods["a_marp"][dataset] - baseline)
                else:
                    a_marp_imp.append(0)
                
                if "dbe" in methods and dataset in methods["dbe"]:
                    dbe_imp.append(methods["dbe"][dataset] - baseline)
                else:
                    dbe_imp.append(0)
                
                if "marc" in methods and dataset in methods["marc"]:
                    marc_imp.append(methods["marc"][dataset] - baseline)
                else:
                    marc_imp.append(0)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Model": models,
            "A-MARP": a_marp_imp,
            "DBE": dbe_imp,
            "MARC": marc_imp
        })
        
        # Melt DataFrame for seaborn
        df_melted = pd.melt(df, id_vars=["Model"], var_name="Method", value_name="Improvement")
        
        # Plot
        sns.barplot(x="Model", y="Improvement", hue="Method", data=df_melted)
        plt.title(f"Performance Improvement Over Standard CoT - {dataset.upper()}")
        plt.xlabel("Model")
        plt.ylabel("Accuracy Improvement (%)")
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.legend(title="Method")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"cross_model_improvement_{dataset}.png"))
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


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze reasoning experiment results")
    parser.add_argument("--results_dir", type=str, default="experiments/results", help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, default="experiments/figures", help="Directory to save visualizations")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = parser.parse_args()
    
    # Load results
    results_map = load_results(args.results_dir)
    
    # Print summary table
    print_summary_table(results_map)
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_boundary_performance(results_map, args.output_dir)
        visualize_cross_model_performance(results_map, args.output_dir)
        visualize_token_efficiency(results_map, args.output_dir)


if __name__ == "__main__":
    main()
