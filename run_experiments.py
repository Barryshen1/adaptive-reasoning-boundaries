"""
Script to run experiments with different reasoning methods
"""
import argparse
import os
import json
import asyncio
import time
from tqdm import tqdm

from methods.a_marp import AMARP
from methods.dbe import DBE
from methods.marc import MARC
from data.loaders.dataset_loaders import get_dataset_loader
from utils.request_tool import MMRequestor


def setup_output_directories():
    """Create necessary output directories"""
    os.makedirs("experiments/results", exist_ok=True)
    os.makedirs("experiments/configs", exist_ok=True)
    os.makedirs("evaluation/results", exist_ok=True)


async def run_experiment(
    method_name,
    dataset_name,
    model_name,
    api_key=None,
    sample_size=100,
    output_path=None,
    config=None
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
        
    Returns:
        Path to results file
    """
    # Set default output path if not provided
    if output_path is None:
        file_name = f"{method_name}_{dataset_name}_{model_name.replace('-', '_')}.jsonl"
        output_path = os.path.join("experiments", "results", file_name)
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = get_dataset_loader(dataset_name, sample_size=sample_size)
    
    # Set up method
    method = None
    if method_name == "a_marp":
        method = AMARP(
            alpha=config.get("alpha", 0.15),
            beta=config.get("beta", 0.08),
            c_max=config.get("c_max", 5)
        )
    elif method_name == "dbe":
        method = DBE(
            gamma=config.get("gamma", 0.12),
            probe_frequency=config.get("probe_frequency", 5),
            probe_set_size=config.get("probe_set_size", 7)
        )
    elif method_name == "marc":
        method = MARC(
            max_communication_rounds=config.get("max_communication_rounds", 5)
        )
        
        # Set up agents for MARC
        agent_configs = config.get("agents", [
            {"type": "planner", "model": model_name},
            {"type": "calculator", "model": model_name},
            {"type": "verifier", "model": model_name}
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
    
    # Set up requestor
    model_type = "openai" if "gpt" in model_name else "anthropic" if "claude" in model_name else "openai"
    requestor = MMRequestor(
        model_type=model_type,
        model_name=model_name,
        api_key=api_key
    )
    
    # Run experiment
    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"Running {method_name} on {dataset_name}")):
        try:
            # Generate prompt based on method
            if method_name == "standard_cot":
                prompt = f"Solve this step by step:\n{item['question']}"
            elif method_name == "a_marp":
                # Get difficulty estimate (simplified)
                difficulty = 1.0
                prompt = method.generate_prompt(item["question"], difficulty, model_name)
            elif method_name == "dbe":
                # For DBE, we'll simulate the interaction sequence
                current_interaction = 0
                prompt, _, _ = method.process_interaction(
                    item["question"], 
                    model_name,
                    current_interaction
                )
            elif method_name == "marc":
                # For MARC, we need a more complex workflow
                # This is a simplified version - in practice, would involve multiple API calls
                collaboration_record = method.solve_task(item["question"])
                prompt = f"Solve this collaborative reasoning task:\n{item['question']}"
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Make API request
            start_time = time.time()
            response = await requestor.request(prompt, temperature=0.7, max_tokens=1024)
            end_time = time.time()
            
            # Save result
            result = {
                "index": item["index"],
                "method": method_name,
                "dataset": dataset_name,
                "model": model_name,
                "pred": response,
                "origin": item,
                "elapsed_time": end_time - start_time
            }
            
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
        config
    )


if __name__ == "__main__":
    asyncio.run(main())
