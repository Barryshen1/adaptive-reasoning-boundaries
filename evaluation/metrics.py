"""
Evaluation metrics for reasoning methods
"""
import re
import numpy as np
from collections import defaultdict


def accuracy(predictions, targets):
    """
    Calculate accuracy
    
    Args:
        predictions: List of predicted values
        targets: List of target values
        
    Returns:
        Accuracy score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def numerical_accuracy(predictions, targets, tolerance=0.01):
    """
    Calculate accuracy for numerical predictions with tolerance
    
    Args:
        predictions: List of predicted values
        targets: List of target values
        tolerance: Acceptable error margin
        
    Returns:
        Accuracy score
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for p, t in zip(predictions, targets):
        try:
            p_val = float(p)
            t_val = float(t)
            if abs(p_val - t_val) <= tolerance:
                correct += 1
        except:
            # If conversion fails, treat as incorrect
            pass
    
    return correct / len(predictions)


def extract_answer(text):
    """
    Extract numerical answer from text
    
    Args:
        text: Text containing the answer
        
    Returns:
        Extracted answer or None
    """
    # Try to find answer after #### marker
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
        # Extract the first number
        matches = re.findall(r'-?\d+\.?\d*', answer_part.replace(",", ""))
        if matches:
            return matches[0]
    
    # Try to extract from the last line
    lines = text.strip().split("\n")
    if lines:
        last_line = lines[-1]
        matches = re.findall(r'-?\d+\.?\d*', last_line.replace(",", ""))
        if matches:
            return matches[-1]
    
    # Try to find any number in the text
    matches = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    if matches:
        return matches[-1]
    
    return None


def boundary_performance(results, boundary_categories):
    """
    Analyze performance across reasoning boundary categories
    
    Args:
        results: List of result dictionaries
        boundary_categories: List of category assignments
        
    Returns:
        Performance metrics by category
    """
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for result, category in zip(results, boundary_categories):
        category_results[category]["total"] += 1
        if result.get("correct", False):
            category_results[category]["correct"] += 1
    
    # Calculate accuracy for each category
    for category in category_results:
        if category_results[category]["total"] > 0:
            category_results[category]["accuracy"] = (
                category_results[category]["correct"] / category_results[category]["total"]
            )
        else:
            category_results[category]["accuracy"] = 0.0
    
    return dict(category_results)


def token_efficiency(results):
    """
    Calculate token efficiency metrics
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Token efficiency metrics
    """
    if not results:
        return {
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "avg_total_tokens": 0,
            "input_token_std": 0,
            "output_token_std": 0
        }
    
    input_tokens = [r.get("input_tokens", 0) for r in results]
    output_tokens = [r.get("output_tokens", 0) for r in results]
    total_tokens = [i + o for i, o in zip(input_tokens, output_tokens)]
    
    return {
        "avg_input_tokens": np.mean(input_tokens),
        "avg_output_tokens": np.mean(output_tokens),
        "avg_total_tokens": np.mean(total_tokens),
        "input_token_std": np.std(input_tokens),
        "output_token_std": np.std(output_tokens)
    }


def adaptation_effectiveness(initial_results, adapted_results):
    """
    Calculate effectiveness of dynamic adaptation
    
    Args:
        initial_results: List of result dictionaries before adaptation
        adapted_results: List of result dictionaries after adaptation
        
    Returns:
        Adaptation effectiveness metrics
    """
    if len(initial_results) != len(adapted_results):
        raise ValueError("Initial and adapted results must have the same length")
    
    if len(initial_results) == 0:
        return {
            "improvement_rate": 0.0,
            "degradation_rate": 0.0,
            "net_improvement": 0.0
        }
    
    initial_correct = [r.get("correct", False) for r in initial_results]
    adapted_correct = [r.get("correct", False) for r in adapted_results]
    
    improvements = sum(1 for i, a in zip(initial_correct, adapted_correct) if not i and a)
    degradations = sum(1 for i, a in zip(initial_correct, adapted_correct) if i and not a)
    
    return {
        "improvement_rate": improvements / len(initial_results),
        "degradation_rate": degradations / len(initial_results),
        "net_improvement": (improvements - degradations) / len(initial_results)
    }


def error_analysis(results, predictions, targets):
    """
    Perform error analysis
    
    Args:
        results: List of result dictionaries
        predictions: List of predictions
        targets: List of targets
        
    Returns:
        Error analysis metrics
    """
    if len(results) != len(predictions) or len(results) != len(targets):
        raise ValueError("Results, predictions, and targets must have the same length")
    
    if len(results) == 0:
        return {
            "error_types": {},
            "error_rate": 0.0
        }
    
    # Define error patterns
    error_patterns = {
        "calculation": r'(\d+\.?\d*)\s*[\+\-\*\/]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
        "planning": r'step|steps|first|then|next|finally',
        "memory": r'mentioned|previously|above|recall|remember',
        "conceptual": r'concept|definition|means|refer|understand'
    }
    
    # Analyze errors
    error_counts = defaultdict(int)
    total_errors = 0
    
    for i, (result, pred, target) in enumerate(zip(results, predictions, targets)):
        if not result.get("correct", False):
            total_errors += 1
            response = result.get("response", "")
            
            # Check error patterns
            error_found = False
            for error_type, pattern in error_patterns.items():
                if re.search(pattern, response, re.IGNORECASE):
                    error_counts[error_type] += 1
                    error_found = True
            
            if not error_found:
                error_counts["other"] += 1
    
    # Calculate error distribution
    error_types = {}
    if total_errors > 0:
        for error_type, count in error_counts.items():
            error_types[error_type] = count / total_errors
    
    return {
        "error_types": dict(error_types),
        "error_rate": total_errors / len(results)
    }
