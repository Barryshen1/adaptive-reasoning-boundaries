"""
Script to run experiments with different reasoning methods
"""
import argparse
import os
import json
import asyncio
import time
import random
import numpy as np
from tqdm import tqdm

from methods.a_marp import AMARP
from methods.dbe import DBE
from methods.marc import MARC
from data.loaders.dataset_loaders import get_dataset_loader
from utils.request_tool import MMRequestor
from utils.tools import estimate_task_difficulty


def setup_output_directories():
    """Create necessary output directories"""
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/configs", exist_ok=True)
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("experiments/figures", exist_ok=True)


async def run_experiment(
    method_name,
    dataset_name,
    model_name,
    api_key=None,
    sample_size=100,
    output_path=None,
    config=None,
    runs=3  # Number of runs with different seeds for stability
):
    """
    Run an experiment with a specific method, dataset, and model
    
    Args:
        method_name: Name of the method (a_marp, dbe, marc, standard_cot)
        dataset_name: Name of the dataset
        model_name: Name of the model
        api_key: API key for model access
        sample_size: Number of samples to evaluate
        output_path: Path to save results
        config: Method configuration
        runs: Number of runs with different seeds
        
    Returns:
        Path to results file
    """
    # Set default output path if not provided
    if output_path is None:
        file_name = f"{method_name}_{dataset_name}_{model_name.replace('-', '_')}.jsonl"
        output_path = os.path.join("experiments", "results", file_name)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = get_dataset_loader(dataset_name, sample_size=sample_size)
    
    # Default configurations based on Section 4.5 of the paper
    default_configs = {
        "a_marp": {
            "alpha": 0.15,
            "beta": 0.08,
            "c_max": 5,
            "dbe": {
                "gamma": 0.12,
                "probe_frequency": 5,
                "probe_set_size": 7
            }
        },
        "dbe": {
            "gamma": 0.12,
            "probe_frequency": 5,
            "probe_set_size": 7
        },
        "marc": {
            "max_communication_rounds": 5
        }
    }
    
    # Merge default config with provided config
    method_config = default_configs.get(method_name, {})
    if config:
        method_config.update(config)
    
    # Set up method
    method = None
    if method_name == "a_marp":
        # Extract DBE-specific parameters
        dbe_params = method_config.get("dbe", {})
        if not dbe_params:
            # Extract DBE parameters from main config if not specified in "dbe" key
            dbe_params = {
                "gamma": method_config.get("gamma", 0.12),
                "probe_frequency": method_config.get("probe_frequency", 5),
                "probe_set_size": method_config.get("probe_set_size", 7)
            }
        
        # Initialize A-MARP with integrated DBE
        method = AMARP(
            alpha=method_config.get("alpha", 0.15),
            beta=method_config.get("beta", 0.08),
            c_max=method_config.get("c_max", 5),
            dbe_params=dbe_params
        )
    elif method_name == "dbe":
        method = DBE(
            gamma=method_config.get("gamma", 0.12),
            probe_frequency=method_config.get("probe_frequency", 5),
            probe_set_size=method_config.get("probe_set_size", 7)
        )
    elif method_name == "marc":
        method = MARC(
            max_communication_rounds=method_config.get("max_communication_rounds", 5)
        )
        
        # Set up agents for MARC based on Section 4.2 of the paper
        agent_configs = method_config.get("agents", [
            {"type": "planner", "model": model_name},
            {"type": "calculator", "model": model_name},
            {"type": "verifier", "model": model_name},
            {"type": "integrator", "model": model_name}
        ])
        
        for i, agent_config in enumerate(agent_configs):
            # Default boundary profiles based on agent type
            boundaries = {
                "planner": {"planning": 10.0, "calculation": 5.0, "working_memory": 7.0},
                "calculator": {"planning": 5.0, "calculation": 15.0, "working_memory": 7.0},
                "verifier": {"planning": 7.0, "calculation": 7.0, "working_memory": 10.0},
                "integrator": {"planning": 8.0, "calculation": 6.0, "working_memory": 12.0}
            }
            
            method.add_agent(
                f"agent_{i}",
                agent_config["type"],
                agent_config.get("boundaries", boundaries.get(agent_config["type"], {})),
                agent_config.get("model", model_name)
            )
    
    # Set up requestor with appropriate API
    model_type = "openai" if any(name in model_name for name in ["gpt", "o1", "o3"]) else \
                "anthropic" if any(name in model_name for name in ["claude"]) else \
                "openai"  # Default to OpenAI
    
    requestor = MMRequestor(
        model_type=model_type,
        model_name=model_name,
        api_key=api_key
    )
    
    # Run multiple times with different seeds for stability
    all_results = []
    for run in range(runs):
        # Set seed for this run
        seed = 42 + run
        np.random.seed(seed)
        random.seed(seed)
        
        # Run experiment for this seed
        print(f"Running experiment with seed {seed} ({run+1}/{runs})")
        results = []
        for i, item in enumerate(tqdm(dataset, desc=f"Running {method_name} on {dataset_name}")):
            try:
                # Generate prompt based on method
                if method_name == "standard_cot":
                    prompt = f"Solve this step by step:\n{item['question']}"
                elif method_name == "a_marp":
                    # Get difficulty estimate
                    difficulty = estimate_task_difficulty(item["question"])
                    
                    # Use the integrated process_with_dbe method
                    result = await method.process_with_dbe(
                        item["question"], 
                        difficulty, 
                        model_name,
                        requestor,
                        i  # Pass the current item index as interaction count
                    )
                    
                    # Extract the prompt
                    prompt = result["prompt"]
                elif method_name == "dbe":
                    # For DBE, we'll simulate the interaction sequence
                    current_interaction = i % method.probe_frequency  # Reset interaction counter periodically
                    prompt, _, _ = method.process_interaction(
                        item["question"], 
                        model_name,
                        current_interaction
                    )
                elif method_name == "marc":
                    # For MARC, we need a more complex workflow
                    collaboration_record = method.solve_task(item["question"])
                    prompt = f"Solve this collaborative reasoning task:\n{item['question']}"
                else:
                    raise ValueError(f"Unknown method: {method_name}")
                
                # Make API request (with retries for API failures)
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        start_time = time.time()
                        response = await requestor.request(prompt, temperature=0.7, max_tokens=1024)
                        end_time = time.time()
                        
                        # If using A-MARP, update the DBE interaction history
                        if method_name == "a_marp" and hasattr(method, "dbe_instance"):
                            # Get the response text
                            response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else ""
                            # Update DBE's interaction history with the response
                            method.dbe_instance.update_interaction_history(prompt, response_text)
                        
                        break
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"API error, retrying ({retry+1}/{max_retries}): {e}")
                            await asyncio.sleep(2 ** retry)  # Exponential backoff
                        else:
                            raise
                
                # Save result
                result = {
                    "index": item["index"],
                    "method": method_name,
                    "dataset": dataset_name,
                    "model": model_name,
                    "pred": response,
                    "origin": item,
                    "elapsed_time": end_time - start_time,
                    "seed": seed,
                    "run": run
                }
                
                # If using A-MARP, store boundary estimates
                if method_name == "a_marp" and hasattr(method, "dbe_instance"):
                    result["boundary_estimates"] = method.dbe_instance.boundary_estimates
                
                # Append to results
                results.append(result)
                
                # Save batch results every 10 items
                if (i + 1) % 10 == 0:
                    with open(output_path, "a", encoding="utf8") as f:
                        for res in results[-10:]:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                            
            except Exception as e:
                print(f"Error processing item {i}: {e}")
        
        # Save any remaining results
        remaining = len(results) % 10
        if remaining > 0:
            with open(output_path, "a", encoding="utf8") as f:
                for res in results[-remaining:]:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
        
        all_results.extend(results)
    
    print(f"Experiment completed. Results saved to {output_path}")
    return output_path


async def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description="Run reasoning experiments")
    parser.add_argument("--method", type=str, required=True, help="Method (a_marp, dbe, marc, standard_cot)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, help="API key (optional, will use environment variable if not provided)")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_path", type=str, help="Custom output path")
    parser.add_argument("--config_path", type=str, help="Path to method configuration")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs with different seeds")
    args = parser.parse_args()
    
    # Set up directories
    setup_output_directories()
    
    # Load configuration
    config = {}
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config = json.load(f)
    
    # Run experiment
    await run_experiment(
        args.method,
        args.dataset,
        args.model,
        args.api_key,
        args.sample_size,
        args.output_path,
        config,
        args.runs
    )


if __name__ == "__main__":
    asyncio.run(main())
