"""
Common utility functions for reasoning boundary experiments
"""
import json
import os
import re
import numpy as np


def read_jsonl(data_path):
    """
    Read data from a JSONL file
    
    Args:
        data_path: Path to JSONL file
        
    Returns:
        List of JSON objects
    """
    input_data = []
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                input_data.append(json.loads(line.strip()))
    else:
        print(f"Missing {data_path}")
    return input_data


def write_jsonl(save_path, save_object, mode="a"):
    """
    Write data to a JSONL file
    
    Args:
        save_path: Path to save to
        save_object: Data to save
        mode: File open mode
    """
    with open(save_path, mode, encoding="utf8") as f:
        for obj in save_object:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def evaluate_expression(expression):
    """
    Safely evaluate a mathematical expression
    
    Args:
        expression: Mathematical expression string
        
    Returns:
        Evaluation result and maximum values
    """
    max_dict = {"plus": 0, "time": 0}
    
    def parse_expression(i):
        value, i = parse_term(i)
        while i < len(expression) and expression[i] in '+-':
            if expression[i] == '+':
                i += 1
                right_value, i = parse_term(i)
                value += right_value
                if abs(value) > abs(max_dict["plus"]):
                    max_dict["plus"] = abs(value)
            elif expression[i] == '-':
                i += 1
                right_value, i = parse_term(i)
                value -= right_value
                
        return value, i
    
    def parse_term(i):
        value, i = parse_factor(i)
        while i < len(expression) and expression[i] in '*/':
            if expression[i] == '*':
                i += 1
                right_value, i = parse_factor(i)
                value *= right_value
                if abs(value) > abs(max_dict["time"]):
                    max_dict["time"] = abs(value)
            elif expression[i] == '/':
                i += 1
                right_value, i = parse_factor(i)
                value /= right_value
        return value, i
    
    def parse_factor(i):
        if expression[i] == '(':
            i += 1  # Skip '('
            value, i = parse_expression(i)
            i += 1  # Skip ')'
        else:
            start_i = i
            while i < len(expression) and (expression[i] == "." or expression[i].isdigit()):
                i += 1
            if "." in expression:
                value = float(expression[start_i:i])
            else:
                value = int(expression[start_i:i])
        return value, i
    
    expression = expression.replace(' ', '')
    if expression.startswith("-"):
        expression = "0" + expression
    value, _ = parse_expression(0)
    return value, max_dict


def get_combined_granularity(origin_data, return_dict=False):
    """
    Calculate the combined reasoning granularity for a data sample
    
    Args:
        origin_data: Original data sample
        return_dict: Whether to return detailed breakdown
        
    Returns:
        Combined granularity value or dictionary with breakdown
    """
    # Parameters based on the paper
    N = 1.6e5
    M = 7.0
    SIGMA = 20000
    
    # Extract equations from the answer
    origin_eqs = [s for s in re.findall(r'<<(.*)?>>', origin_data["answer"])]
    
    # Extract operations
    operation_list = [operation for eq1 in origin_eqs for operation in re.findall(r'[\+\-\*/]', eq1.split("=")[0])]
    
    # Find maximum calculation complexity
    max_time = 0
    for eq0 in origin_eqs:
        try:
            _, max_dict = evaluate_expression(eq0.split("=")[0])
            if max_time < max_dict["time"]:
                max_time = max_dict["time"]
        except:
            # Skip problematic expressions
            pass
    
    calculate_granularity = max_time
    
    # Calculate planning granularity
    if len(operation_list) == len(origin_eqs):
        plan_granularity = len([x for x in origin_eqs if not x.strip("0.").startswith("0")])
    else:
        plan_granularity = len(operation_list)
    
    # Calculate combined granularity using the combination formula from the paper
    combined = 1/(M/plan_granularity + N/(calculate_granularity + SIGMA))
    
    if return_dict:
        return {
            "plan_granularity": plan_granularity,
            "calculate_granularity": calculate_granularity,
            "combined_granularity": combined
        }
    
    return combined


def estimate_task_difficulty(task):
    """
    Estimate the difficulty of a reasoning task
    
    Args:
        task: Reasoning task text
        
    Returns:
        Estimated difficulty score (1-5 scale)
    """
    # Count words as basic complexity measure
    word_count = len(task.split())
    
    # Count special keywords that indicate complexity
    complexity_indicators = [
        "calculate", "compute", "solve", "find", "prove",
        "explain", "analyze", "compare", "evaluate", "determine"
    ]
    
    indicator_count = sum(1 for indicator in complexity_indicators if indicator in task.lower())
    
    # Detect numerical values as indicators of calculation complexity
    numbers = re.findall(r'\d+\.?\d*', task)
    number_count = len(numbers)
    
    # Check for large numbers (indicating calculation complexity)
    large_numbers = sum(1 for num in numbers if len(num.replace('.', '')) > 3)
    
    # Detect multi-step reasoning indicators
    step_indicators = [
        "steps", "first", "then", "next", "finally", "after",
        "step", "sequence", "process", "method"
    ]
    
    step_count = sum(1 for indicator in step_indicators if indicator in task.lower())
    
    # Detect logical operators as indicators of logical complexity
    logical_operators = ["if", "and", "or", "not", "all", "some", "none", "every", "any"]
    logical_count = sum(1 for op in logical_operators if f" {op} " in f" {task.lower()} ")
    
    # Base difficulty calculation
    difficulty = (
        0.5 +                          # Base difficulty
        (word_count / 100) * 0.5 +     # Length factor
        (indicator_count * 0.2) +      # Complexity indicators
        (number_count * 0.1) +         # Number of numerical values
        (large_numbers * 0.2) +        # Large numbers (calculation difficulty)
        (step_count * 0.3) +           # Multi-step indicators
        (logical_count * 0.15)         # Logical complexity
    )
    
    return min(5.0, difficulty)  # Cap at 5.0 (very difficult)


def categorize_boundary(granularity, thresholds=None):
    """
    Categorize a granularity value into a reasoning boundary category
    
    Args:
        granularity: Granularity value
        thresholds: Dictionary with K and K2 threshold values
        
    Returns:
        Boundary category (CFRB, PFRB, or CIRB) and accuracy range
    """
    # Default thresholds from paper
    if thresholds is None:
        thresholds = {"K": 0.106, "K2": 0.425}
    
    K = thresholds.get("K", 0.106)
    K2 = thresholds.get("K2", 0.425)
    
    if granularity <= K:
        return "CFRB", ">90%"
    elif granularity <= K2:
        return "PFRB", "10%-90%"
    else:
        return "CIRB", "<10%"


def calculate_boundary_combination(boundaries, scaling_factors=None):
    """
    Calculate combined boundary using the combination law
    
    Args:
        boundaries: Dictionary of boundaries by dimension
        scaling_factors: Dictionary of N and b scaling factors
        
    Returns:
        Combined boundary value
    """
    if not boundaries:
        return 0.0
    
    # Default scaling factors from paper
    if scaling_factors is None:
        scaling_factors = {
            "calculation": {"N": 1.6e5, "b": 20000},
            "planning": {"N": 7.0, "b": 0.0},
            "working_memory": {"N": 100.0, "b": 1.0}
        }
    
    # Calculate the sum term in the combination law
    sum_term = 0.0
    valid_dimensions = 0
    
    for dimension, boundary in boundaries.items():
        if dimension in scaling_factors:
            N_i = scaling_factors[dimension]["N"]
            b_i = scaling_factors[dimension]["b"]
            
            # Avoid division by zero or negative values
            denominator = max(0.001, boundary - b_i)
            sum_term += N_i / denominator
            valid_dimensions += 1
    
    # Calculate combined boundary using the formula from the paper
    if valid_dimensions > 1 and sum_term > 0:
        return 1.0 / ((valid_dimensions - 1) * sum_term)
    elif valid_dimensions == 1:
        # With only one dimension, return its boundary
        return list(boundaries.values())[0]
    else:
        return 0.0


def extract_calculation_steps(text):
    """
    Extract calculation steps from a reasoning text
    
    Args:
        text: Reasoning text
        
    Returns:
        List of calculation steps
    """
    # Extract calculations between << >> markers
    calculations = re.findall(r'<<(.*?)>>', text)
    
    # Extract other calculation steps (expressions with = sign)
    equations = re.findall(r'(\d+[+\-*/]\d+\s*=\s*\d+\.?\d*)', text)
    
    # Combine and clean up
    steps = []
    for calc in calculations + equations:
        if "=" in calc:
            left, right = calc.split("=")
            steps.append({"expression": left.strip(), "result": right.strip()})
    
    return steps


def extract_reasoning_structure(text):
    """
    Extract reasoning structure from text
    
    Args:
        text: Reasoning text
        
    Returns:
        Dictionary with reasoning structure information
    """
    # Count the number of steps
    step_markers = re.findall(r'step\s*\d+|first|second|third|next|finally', text.lower())
    num_steps = len(step_markers)
    
    # Detect structure markers
    structure_markers = [
        "let's", "we need to", "first", "then", "next", "after", "finally",
        "to solve", "to find", "to calculate", "we can", "we must"
    ]
    
    structure_count = sum(1 for marker in structure_markers if marker in text.lower())
    
    # Detect verification steps
    verification_markers = [
        "check", "verify", "confirm", "validate", "make sure", "substitut"
    ]
    
    verification_count = sum(1 for marker in verification_markers if marker in text.lower())
    
    return {
        "num_steps": num_steps,
        "has_structure": structure_count > 0,
        "structure_score": min(1.0, structure_count / 5),
        "has_verification": verification_count > 0,
        "verification_score": min(1.0, verification_count / 3)
    }
