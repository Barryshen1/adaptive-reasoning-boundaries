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
        _, max_dict = evaluate_expression(eq0.split("=")[0])
        if max_time < max_dict["time"]:
            max_time = max_dict["time"]
    
    calculate_granularity = max_time
    
    # Calculate planning granularity
    if len(operation_list) == len(origin_eqs):
        plan_granularity = len([x for x in origin_eqs if not x.strip("0.").startswith("0")])
    else:
        plan_granularity = len(operation_list)
    
    # Calculate combined granularity
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
        Estimated difficulty score
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
    number_count = len(re.findall(r'\d+\.?\d*', task))
    
    # Base difficulty on these factors
    difficulty = 0.5 + (word_count / 100) + (indicator_count * 0.2) + (number_count * 0.1)
    
    return min(5.0, difficulty)  # Cap at 5.0 (very difficult)


def categorize_boundary(granularity, thresholds):
    """
    Categorize a granularity value into a reasoning boundary category
    
    Args:
        granularity: Granularity value
        thresholds: Dictionary with K and K2 threshold values
        
    Returns:
        Boundary category (CFRB, PFRB, or CIRB)
    """
    K = thresholds.get("K", 0.106)
    K2 = thresholds.get("K2", 0.425)
    
    if granularity <= K:
        return "CFRB", ">90%"
    elif granularity <= K2:
        return "PFRB", "10%-90%"
    else:
        return "CIRB", "<10%"
