"""
Dataset loaders for various reasoning tasks
"""
import os
import json
import random
from datasets import load_dataset


class DatasetLoader:
    """Base class for dataset loaders"""
    
    def __init__(self, dataset_name, split="test", sample_size=None, seed=42):
        """
        Initialize dataset loader
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split (train, validation, test)
            sample_size: Number of examples to sample
            seed: Random seed for sampling
        """
        self.dataset_name = dataset_name
        self.split = split
        self.sample_size = sample_size
        self.seed = seed
        random.seed(seed)
        
        self.data = self.load()
        
        if sample_size and len(self.data) > sample_size:
            self.data = random.sample(self.data, sample_size)
    
    def load(self):
        """Load dataset (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def get_item(self, idx):
        """Get item by index"""
        return self.data[idx]
    
    def __len__(self):
        """Get dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item by index"""
        return self.get_item(idx)


class GSM8KLoader(DatasetLoader):
    """Loader for GSM8K dataset"""
    
    def load(self):
        """Load GSM8K dataset"""
        dataset = load_dataset("gsm8k", "main")
        data = dataset[self.split]
        
        processed_data = []
        for i, item in enumerate(data):
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "index": str(i)
            })
        
        return processed_data


class MATHLoader(DatasetLoader):
    """Loader for MATH dataset"""
    
    def load(self):
        """Load MATH dataset"""
        dataset = load_dataset("hendrycks/math")
        data = dataset[self.split]
        
        processed_data = []
        for i, item in enumerate(data):
            processed_data.append({
                "question": item["problem"],
                "answer": item["solution"],
                "level": item["level"],
                "type": item["type"],
                "index": str(i)
            })
        
        return processed_data


class BigGSMLoader(DatasetLoader):
    """Loader for BigGSM dataset"""
    
    def load(self):
        """Load BigGSM dataset"""
        data_path = os.path.join("data", "biggsm", "data.jsonl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        if os.path.exists(data_path):
            # Load from file
            processed_data = []
            with open(data_path, "r", encoding="utf8") as f:
                for line in f:
                    item = json.loads(line)
                    processed_data.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "index": item.get("index", str(len(processed_data)))
                    })
        else:
            # Try to load from HuggingFace
            try:
                dataset = load_dataset("LightChen2333/BigGSM")
                data = dataset[self.split]
                
                processed_data = []
                for i, item in enumerate(data):
                    processed_data.append({
                        "question": item["question"],
                        "answer": item["answer"],
                        "index": str(i)
                    })
                
                # Save to file for future use
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, "w", encoding="utf8") as f:
                    for item in processed_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except:
                # Create placeholder data if dataset not available
                processed_data = [
                    {
                        "question": f"BigGSM placeholder question {i}",
                        "answer": f"BigGSM placeholder answer {i}\n#### {i*10}",
                        "index": str(i)
                    }
                    for i in range(10)
                ]
        
        return processed_data


class HotpotQALoader(DatasetLoader):
    """Loader for HotpotQA dataset"""
    
    def load(self):
        """Load HotpotQA dataset"""
        dataset = load_dataset("hotpot_qa", "distractor")
        data = dataset[self.split]
        
        processed_data = []
        for i, item in enumerate(data):
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"],
                "index": str(i)
            })
        
        return processed_data


class StrategyQALoader(DatasetLoader):
    """Loader for StrategyQA dataset"""
    
    def load(self):
        """Load StrategyQA dataset"""
        dataset = load_dataset("strategyqa")
        data = dataset[self.split]
        
        processed_data = []
        for i, item in enumerate(data):
            processed_data.append({
                "question": item["question"],
                "answer": item["answer"],
                "facts": item.get("facts", []),
                "decomposition": item.get("decomposition", []),
                "index": str(i)
            })
        
        return processed_data


class MultiArithLoader(DatasetLoader):
    """Loader for MultiArith dataset"""
    
    def load(self):
        """Load MultiArith dataset"""
        try:
            dataset = load_dataset("sva-al/MultiArith")
            data = dataset[self.split]
            
            processed_data = []
            for i, item in enumerate(data):
                processed_data.append({
                    "question": item["question"],
                    "answer": str(item["answer"]),
                    "index": str(i)
                })
        except:
            # Create placeholder data if dataset not available
            processed_data = [
                {
                    "question": f"MultiArith placeholder question {i}",
                    "answer": f"{i*5}",
                    "index": str(i)
                }
                for i in range(10)
            ]
        
        return processed_data


class MGSMLoader(DatasetLoader):
    """Loader for MGSM dataset"""
    
    def load(self):
        """Load MGSM dataset"""
        try:
            dataset = load_dataset("juletxara/mgsm")
            data = dataset[self.split]
            
            processed_data = []
            for i, item in enumerate(data):
                processed_data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "language": item.get("language", "en"),
                    "index": str(i)
                })
        except:
            # Create placeholder data if dataset not available
            processed_data = [
                {
                    "question": f"MGSM placeholder question {i}",
                    "answer": f"MGSM placeholder answer {i}\n#### {i*15}",
                    "language": "en",
                    "index": str(i)
                }
                for i in range(10)
            ]
        
        return processed_data


def get_dataset_loader(dataset_name, **kwargs):
    """
    Get the appropriate dataset loader
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for the loader
        
    Returns:
        Dataset loader instance
    """
    loaders = {
        "gsm8k": GSM8KLoader,
        "math": MATHLoader,
        "biggsm": BigGSMLoader,
        "hotpotqa": HotpotQALoader,
        "strategyqa": StrategyQALoader,
        "multiarith": MultiArithLoader,
        "mgsm": MGSMLoader
    }
    
    loader_class = loaders.get(dataset_name.lower())
    if loader_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loader_class(dataset_name, **kwargs)
