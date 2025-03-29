"""
Evaluation script for reasoning methods
"""
import argparse
import os
import json
import numpy as np
import tiktoken
from prettytable import PrettyTable
from tqdm import tqdm

import sys
sys.path.append('/workspace/adaptive-reasoning-boundaries')
from utils.request_tool import RequestOutput
from utils.tools import get_combined_granularity, categorize_boundary
from evaluation.metrics import extract_answer, boundary_performance, token_efficiency
from evaluation.metrics import adaptation_effectiveness, reasoning_path_quality


# Parameter dictionary for different methods
PARAM_DICT = {
    "standard_cot": {
        "K": 0.106,
        "K2": 0.425,
        "mode": "nl",
        "result_path": "experiments/results/standard_cot_{dataset}_{model}.jsonl"
    },
    "marp": {  # Added original MARP method
        "K": 0.106,
        "K2": 0.425,
        "mode": "nl",
        "result_path": "experiments/results/marp_{dataset}_{model}.jsonl"
    },
    "a_marp": {
        "K": 0.12,
        "K2": 0.53,
        "mode": "nl",
        "result_path": "experiments/results/a_marp_{dataset}_{model}.jsonl"
    },
    "dbe": {
        "K": 0.13,
        "K2": 0.81,
        "mode": "nl",
        "result_path": "experiments/results/dbe_{dataset}_{model}.jsonl"
    },
    "marc": {
        "K": 0.106,
        "K2": 0.50,
        "mode": "nl",
        "result_path": "experiments/results/marc_{dataset}_{model}.jsonl",
        "dbe_params": {
            "gamma": 0.12,
            "probe_frequency": 5,
            "probe_set_size": 7
        }
    }
}


def evaluate_method(result_path, K, K2, mode="nl", verbose=True):
    """
    Evaluate the performance of a reasoning method
    
    Args:
        result_path: Path to results file
        K: CFRB/PFRB threshold
        K2: PFRB/CIRB threshold
        mode: Evaluation mode
        verbose: Whether to print detailed results
        
    Returns:
        Evaluation metrics
    """
    # Check if file exists
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Results file not found: {result_path}")
    
    # Load results
    response_list = RequestOutput(result_path)
    
    # Initialize statistics
    token_num = 0
    input_token_num = 0
    enc = tiktoken.encoding_for_model("gpt-4")
    acc = {">90%": {"correct": 0, "total": 0}, "10%~90%": {"correct": 0, "total": 0}, "<10%": {"correct": 0, "total": 0}}
    
    # For MARC, track additional statistics about DBE integration
    is_marc = "marc" in result_path
    marc_dbe_stats = {
        "boundary_updates": 0,
        "avg_boundary_change": {},
        "agent_count": 0,
        "collaboration_count": 0
    }
    
    # Evaluate each result
    for idx in tqdm(range(len(response_list)), desc="Evaluating", disable=not verbose):
        try:
            # Calculate granularity
            granularity = get_combined_granularity(response_list.get_origin_input(idx))
            
            # Count tokens
            input_token_num += len(enc.encode(response_list.data[idx]["pred"][0]["content"][0]["text"]))
            token_num += len(enc.encode(response_list.get_last_pred_text(idx)))
            
            # Determine boundary category
            if granularity <= K:
                granularity_key = ">90%"
            elif granularity > K and granularity <= K2:
                granularity_key = "10%~90%"
            elif granularity > K2:
                granularity_key = "<10%"
                
            # Check correctness
            correct = response_list.judge_correct(idx, mode=mode)
            if correct:
                acc[granularity_key]["correct"] += 1
            acc[granularity_key]["total"] += 1
            
            # Track MARC-specific statistics
            if is_marc and "boundary_updates" in response_list.data[idx]:
                marc_dbe_stats["boundary_updates"] += 1
                
                # Count number of agents
                if "collaboration_record" in response_list.data[idx]:
                    if "agents" in response_list.data[idx]["collaboration_record"]:
                        marc_dbe_stats["agent_count"] = max(
                            marc_dbe_stats["agent_count"],
                            len(response_list.data[idx]["collaboration_record"]["agents"])
                        )
                    
                    # Count collaborations
                    if "phases" in response_list.data[idx]["collaboration_record"]:
                        for phase in response_list.data[idx]["collaboration_record"]["phases"]:
                            if phase.get("phase") == "processing" and "rounds" in phase:
                                for round_data in phase["rounds"]:
                                    if "actions" in round_data:
                                        for action in round_data["actions"]:
                                            if action.get("action") == "process_collaborative_subtask":
                                                marc_dbe_stats["collaboration_count"] += 1
                
                # Track boundary changes
                for agent_id, updates in response_list.data[idx]["boundary_updates"].items():
                    for dim in ["calculation", "planning", "working_memory"]:
                        if dim not in marc_dbe_stats["avg_boundary_change"]:
                            marc_dbe_stats["avg_boundary_change"][dim] = []
                        
                        # Calculate relative change from initial to final if available
                        if "initial" in updates and f"round_{len(updates)-2}" in updates:
                            initial = updates["initial"].get(dim, 0)
                            final = updates[f"round_{len(updates)-2}"].get(dim, 0)
                            if initial > 0:
                                change = (final - initial) / initial
                                marc_dbe_stats["avg_boundary_change"][dim].append(change)
            
        except Exception as e:
            if verbose:
                print(f"Error evaluating result {idx}: {e}")
    
    # Calculate overall statistics
    total = sum(acc[key]["total"] for key in acc)
    correct = sum(acc[key]["correct"] for key in acc)
    
    # Create results dictionary
    results = {
        "overall_accuracy": round(correct/total * 100, 2) if total > 0 else 0,
        "avg_input_tokens": round(input_token_num/total, 2) if total > 0 else 0,
        "avg_output_tokens": round(token_num/total, 2) if total > 0 else 0,
        "boundary_performance": {}
    }
    
    # Add boundary-specific accuracy
    for key in acc:
        if acc[key]["total"] > 0:
            results["boundary_performance"][key] = {
                "accuracy": round(acc[key]["correct"]/acc[key]["total"] * 100, 2),
                "samples": acc[key]["total"]
            }
        else:
            results["boundary_performance"][key] = {
                "accuracy": "-",
                "samples": 0
            }
    
    # Add MARC-DBE integration statistics
    if is_marc:
        results["marc_dbe_integration"] = {
            "boundary_updates_count": marc_dbe_stats["boundary_updates"],
            "agent_count": marc_dbe_stats["agent_count"],
            "collaboration_count": marc_dbe_stats["collaboration_count"],
        }
        
        # Calculate average boundary changes
        for dim, changes in marc_dbe_stats["avg_boundary_change"].items():
            if changes:
                results["marc_dbe_integration"][f"avg_{dim}_change"] = round(sum(changes) / len(changes) * 100, 2)
            else:
                results["marc_dbe_integration"][f"avg_{dim}_change"] = 0
    
    # Print results table if verbose
    if verbose:
        table = PrettyTable()
        table.field_names = ["Granularity", "Accuracy", "Samples", "Input Tokens", "Output Tokens"]
        
        for key in [">90%", "10%~90%", "<10%"]:
            if acc[key]["total"] > 0:
                table.add_row([
                    key,
                    f"{round(acc[key]['correct']/acc[key]['total'] * 100, 2)}%",
                    acc[key]["total"],
                    "-",
                    "-"
                ], divider=key == "<10%")
            else:
                table.add_row([
                    key,
                    "-",
                    0,
                    "-",
                    "-"
                ], divider=key == "<10%")
                
        table.add_row([
            "All", 
            f"{round(correct/total * 100, 2)}%" if total > 0 else "-", 
            total,
            round(input_token_num/total, 2) if total > 0 else "-", 
            round(token_num/total, 2) if total > 0 else "-"
        ])
        
        print(table)
        
        # Print MARC-DBE integration statistics if applicable
        if is_marc:
            print("\nMARC-DBE Integration Statistics:")
            dbe_table = PrettyTable()
            dbe_table.field_names = ["Metric", "Value"]
            
            dbe_table.add_row(["Agent Count", marc_dbe_stats["agent_count"]])
            dbe_table.add_row(["Boundary Updates Count", marc_dbe_stats["boundary_updates"]])
            dbe_table.add_row(["Collaborative Subtasks", marc_dbe_stats["collaboration_count"]])
            
            for dim, changes in marc_dbe_stats["avg_boundary_change"].items():
                if changes:
                    dbe_table.add_row([
                        f"Avg {dim.capitalize()} Boundary Change", 
                        f"{round(sum(changes) / len(changes) * 100, 2)}%"
                    ])
            
            print(dbe_table)
    
    return results


def compare_methods(methods, dataset_name, model_name, verbose=True):
    """
    Compare multiple reasoning methods
    
    Args:
        methods: List of method names
        dataset_name: Dataset name for display
        model_name: Model name to evaluate with
        verbose: Whether to print detailed results
        
    Returns:
        Comparison results
    """
    results = {}
    
    for method in methods:
        if method not in PARAM_DICT:
            print(f"Warning: Unknown method {method}. Skipping.")
            continue
        
        params = PARAM_DICT[method]
        result_path = params["result_path"].format(dataset=dataset_name, model=model_name)
        
        try:
            method_results = evaluate_method(
                result_path,
                params["K"],
                params["K2"],
                params["mode"],
                verbose=False
            )
            results[method] = method_results
        except FileNotFoundError as e:
            print(f"Error evaluating {method}: {e}")
    
    if verbose and results:
        # Print comparison table
        table = PrettyTable()
        table.field_names = ["Method", "Overall Accuracy", "CFRB Accuracy", "PFRB Accuracy", "CIRB Accuracy", "Avg. Tokens"]
        
        for method, res in results.items():
            table.add_row([
                method,
                f"{res['overall_accuracy']}%",
                f"{res['boundary_performance']['>90%']['accuracy']}%" if res['boundary_performance']['>90%']['accuracy'] != '-' else '-',
                f"{res['boundary_performance']['10%~90%']['accuracy']}%" if res['boundary_performance']['10%~90%']['accuracy'] != '-' else '-',
                f"{res['boundary_performance']['<10%']['accuracy']}%" if res['boundary_performance']['<10%']['accuracy'] != '-' else '-',
                res['avg_input_tokens'] + res['avg_output_tokens']
            ])
        
        print(f"\nResults for {dataset_name} with {model_name}:")
        print(table)
        
        # Print MARC-DBE integration comparison if MARC is included
        if "marc" in results:
            print("\nMARC-DBE Integration:")
            if "marc_dbe_integration" in results["marc"]:
                dbe_stats = results["marc"]["marc_dbe_integration"]
                dbe_table = PrettyTable()
                dbe_table.field_names = ["Metric", "Value"]
                
                for metric, value in dbe_stats.items():
                    dbe_table.add_row([
                        metric.replace("_", " ").title(),
                        f"{value}%" if "change" in metric else value
                    ])
                
                print(dbe_table)
    
    return results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate reasoning methods")
    parser.add_argument("--method", type=str, default="all", help="Method to evaluate (all, standard_cot, marp, a_marp, dbe, marc)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--result_path", type=str, help="Custom result path")
    parser.add_argument("--K", type=float, help="Custom K threshold")
    parser.add_argument("--K2", type=float, help="Custom K2 threshold")
    parser.add_argument("--mode", type=str, default="nl", help="Evaluation mode (nl, tool, pot)")
    args = parser.parse_args()
    
    if args.method.lower() == "all":
        compare_methods(["standard_cot", "marp", "a_marp", "dbe", "marc"], args.dataset, args.model)
    elif args.method in PARAM_DICT:
        params = PARAM_DICT[args.method]
        result_path = args.result_path or params["result_path"].format(dataset=args.dataset, model=args.model)
        evaluate_method(
            result_path,
            params["K"],
            params["K2"],
            params["mode"]
        )
    else:
        # Custom evaluation
        if not args.result_path:
            raise ValueError("Must provide --result_path for custom evaluation")
        if not args.K:
            raise ValueError("Must provide --K for custom evaluation")
        if not args.K2:
            raise ValueError("Must provide --K2 for custom evaluation")
        
        evaluate_method(
            args.result_path,
            args.K,
            args.K2,
            args.mode
        )


if __name__ == "__main__":
    main()
