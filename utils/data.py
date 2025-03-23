"""
Data loading and processing utilities
"""
import json
import random
import re
from copy import deepcopy

random.seed(42)


class GSM8KData:
    """
    Utility class for GSM8K dataset handling
    """
    def __init__(self, obj) -> None:
        self.obj = obj
    
    def get_answer(self):
        """Extract numerical answer from the response"""
        res_str = self.obj["answer"].replace(",", "").strip(".").split("\n#### ")[-1]
        try:
            return round(float(res_str), 2)
        except:
            return -1
    
    def get_text_answer(self):
        """Get the full textual answer"""
        return self.obj["answer"]
    
    def extract_equation(self, obj):
        """Extract equations from the answer"""
        exp_list = [s for s in re.findall(r'<<(.*)?>>', obj["answer"])]
        equation_list = []
        obj["operation"] = {"+": 0, "-": 0, "*": 0, "/": 0}
        for exp in exp_list:
            exp = exp.strip(".0").strip(".00")
            ans = exp.split("=")[-1].strip()
            exp = exp.split("=")[0]
            operations = re.findall(r"\+|\-|\*|\/", exp)
            for operation in operations:
                obj["operation"][operation] += 1
            if ans == "":
                ans = "0"
            equation_list.append({"func": exp, "ans": ans})
        return obj, equation_list
    
    def __str__(self) -> str:
        return json.dumps(self.obj)


class MATHData:
    """
    Utility class for MATH dataset handling
    """
    def __init__(self, obj) -> None:
        self.obj = obj
    
    def get_problem(self):
        """Get the problem text"""
        return self.obj.get("problem", "")
    
    def get_answer(self):
        """Get the answer"""
        return self.obj.get("answer", "")
    
    def get_level(self):
        """Get the difficulty level"""
        return self.obj.get("level", "")
    
    def get_type(self):
        """Get the problem type"""
        return self.obj.get("type", "")
    
    def __str__(self) -> str:
        return json.dumps(self.obj)


class HotpotQAData:
    """
    Utility class for HotpotQA dataset handling
    """
    def __init__(self, obj) -> None:
        self.obj = obj
    
    def get_question(self):
        """Get the question"""
        return self.obj.get("question", "")
    
    def get_answer(self):
        """Get the answer"""
        return self.obj.get("answer", "")
    
    def get_supporting_facts(self):
        """Get supporting facts"""
        return self.obj.get("supporting_facts", [])
    
    def get_context(self):
        """Get the context paragraphs"""
        return self.obj.get("context", [])
    
    def __str__(self) -> str:
        return json.dumps(self.obj)


class StrategyQAData:
    """
    Utility class for StrategyQA dataset handling
    """
    def __init__(self, obj) -> None:
        self.obj = obj
    
    def get_question(self):
        """Get the question"""
        return self.obj.get("question", "")
    
    def get_answer(self):
        """Get the boolean answer"""
        return self.obj.get("answer", False)
    
    def get_facts(self):
        """Get the facts needed for reasoning"""
        return self.obj.get("facts", [])
    
    def get_decomposition(self):
        """Get the reasoning decomposition"""
        return self.obj.get("decomposition", [])
    
    def __str__(self) -> str:
        return json.dumps(self.obj)


def load_dataset(dataset_name, split="train", sample_size=None):
    """
    Load and preprocess a dataset
    
    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split (train, validation, test)
        sample_size: Number of examples to sample (None for all)
        
    Returns:
        Processed dataset
    """
    if dataset_name.lower() == "gsm8k":
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main")
        data = dataset[split]
        
        processed_data = []
        for item in data:
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "index": str(len(processed_data))
            })
    
    elif dataset_name.lower() == "math":
        from datasets import load_dataset
        dataset = load_dataset("hendrycks/math")
        data = dataset[split]
        
        processed_data = []
        for item in data:
            processed_data.append({
                "question": item["problem"],
                "answer": item["solution"],
                "level": item["level"],
                "type": item["type"],
                "index": str(len(processed_data))
            })
    
    elif dataset_name.lower() == "biggsm":
        # For BigGSM, we need to load it from a jsonl file
        # Placeholder implementation
        import json
        
        try:
            processed_data = []
            with open(f"data/biggsm/data.jsonl", "r") as f:
                for line in f:
                    item = json.loads(line)
                    processed_data.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "index": item.get("index", str(len(processed_data)))
                    })
        except FileNotFoundError:
            # If file doesn't exist, generate placeholder data
            processed_data = [
                {
                    "question": f"BigGSM placeholder question {i}",
                    "answer": f"BigGSM placeholder answer {i}\n#### {i*10}",
                    "index": str(i)
                }
                for i in range(10)
            ]
    
    elif dataset_name.lower() == "hotpotqa":
        from datasets import load_dataset
        dataset = load_dataset("hotpot_qa", "distractor")
        data = dataset[split]
        
        processed_data = []
        for item in data:
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"],
                "index": str(len(processed_data))
            })
    
    elif dataset_name.lower() == "strategyqa":
        from datasets import load_dataset
        dataset = load_dataset("strategyqa")
        data = dataset[split]
        
        processed_data = []
        for item in data:
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "facts": item.get("facts", []),
                "decomposition": item.get("decomposition", []),
                "index": str(i)
            })
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample if requested
    if sample_size and len(processed_data) > sample_size:
        processed_data = random.sample(processed_data, sample_size)
    
    return processed_data


def get_data_wrapper(dataset_name):
    """
    Get the appropriate data wrapper class for a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Data wrapper class
    """
    dataset_mapping = {
        "gsm8k": GSM8KData,
        "math": MATHData,
        "biggsm": GSM8KData,
        "hotpotqa": HotpotQAData,
        "strategyqa": StrategyQAData
    }
    
    return dataset_mapping.get(dataset_name.lower(), GSM8KData)
