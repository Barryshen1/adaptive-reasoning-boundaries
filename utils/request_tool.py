"""
Utilities for making API requests to language models
"""
import asyncio
import json
import os
import re
from copy import deepcopy


def judge_error(pred):
    """
    Check if a prediction is valid
    
    Args:
        pred: Prediction to check
        
    Returns:
        True if valid, False otherwise
    """
    try:
        float(pred)
    except:
        return False
    return True


class RequestOutput:
    """
    Class for handling request outputs
    """
    def __init__(self, load_path, auto_index=True) -> None:
        """
        Initialize with output data
        
        Args:
            load_path: Path to load data from
            auto_index: Whether to automatically assign indices
        """
        temp_list = []
        if auto_index:
            for i, temp in enumerate(self._read_jsonl(load_path)):
                temp["index"] = str(i)
                temp_list.append(temp)
            self.data = sorted(self._read_jsonl(load_path), key=lambda x: int(x["index"]))
        else:
            self.data = self._read_jsonl(load_path)
    
    def _read_jsonl(self, data_path):
        """
        Read data from a JSONL file
        
        Args:
            data_path: Path to JSONL file
            
        Returns:
            List of parsed JSON objects
        """
        input_data = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf8") as f:
                for line in f:
                    input_data.append(json.loads(line.strip()))
        else:
            print(f"Missing {data_path}")
        return input_data
    
    def save(self, save_path):
        """
        Save data to a JSONL file
        
        Args:
            save_path: Path to save to
        """
        self._write_jsonl(save_path, self.data, mode="w")
    
    def _write_jsonl(self, save_path, save_object, mode="a"):
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
    
    def get_last_pred_text(self, index):
        """
        Get the text of the last prediction
        
        Args:
            index: Data index
            
        Returns:
            Prediction text
        """
        return self.data[index]["pred"][-1]["content"][0]["text"]
    
    def get_origin_input(self, index):
        """
        Get the original input
        
        Args:
            index: Data index
            
        Returns:
            Original input
        """
        return self.data[index]["origin"]
    
    def search_by_question(self, question):
        """
        Search for an index by question
        
        Args:
            question: Question to search for
            
        Returns:
            Matching index or None
        """
        for i, d in enumerate(self.data):
            if d["origin"]["question"] == question:
                return i
        return None
    
    def __len__(self):
        """Get number of data items"""
        return len(self.data)
    
    def get_pred_answer(self, idx):
        """
        Get the predicted answer
        
        Args:
            idx: Data index
            
        Returns:
            Predicted answer
        """
        pred_text = self.get_last_pred_text(idx)
        
        # Try to find answer after #### marker
        if "####" in pred_text:
            answer_part = pred_text.split("####")[-1].strip()
            # Extract the first number
            matches = re.findall(r'-?\d+\.?\,?\d*', answer_part.replace(",", ""))
            if matches:
                return matches[0]
        
        # Try other extraction methods
        pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', pred_text.replace(",", "").strip(".").split("=")[-1])]
        if len(pred_list) == 0:
            return -1
        else:
            return pred_list[-1]
    
    def get_parsed_pred_answer(self, idx):
        """
        Get the parsed predicted answer (for tool-based methods)
        
        Args:
            idx: Data index
            
        Returns:
            Parsed predicted answer
        """
        pred_str = self.get_last_pred_text(idx)
        if "var" not in pred_str or "<<" not in pred_str:
            pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', pred_str.replace(",", "").strip(".").split("=")[-1])]
            if len(pred_list) == 0:
                pred1 = -1
            else:
                pred1 = pred_list[-1]
            return pred1
        else:
            try:
                eqs = [s for s in re.findall(r'<<(.*?)>>', pred_str) if "=" in s]
                eqs = sorted(eqs, key=lambda x: int(x.split("=")[0].strip("var")))
                var_list = {eq.split("=")[0]: None for eq in eqs}
                
                for eq in eqs:
                    if "=" in eq:
                        func_str = eq.split("=")[1]
                        for var in var_list:
                            if var_list[var] is not None:
                                func_str = func_str.replace(var, str(var_list[var]))
                        if var_list[eq.split("=")[0]] is None:
                            try:
                                var_list[eq.split("=")[0]] = eval(func_str)
                            except:
                                return -1
                    elif "var" in eq:
                        pred_str += "#### " + eq
                        
                if "####" in pred_str:
                    var_key = pred_str.split("####")[-1].strip().strip(".").replace("<", "").replace(">", "")
                    if var_key in var_list:
                        return var_list[var_key]
                        
                last_var = var_list[list(var_list.keys())[-1]]
                if last_var is None:
                    last_var = -1
            except:
                return -1
                
            return last_var
    
    def get_program_answer(self, idx):
        """
        Get the program-derived answer (for PoT methods)
        
        Args:
            idx: Data index
            
        Returns:
            Program-derived answer
        """
        pred_str = self.get_last_pred_text(idx)
        if "def " not in pred_str or "```" not in pred_str:
            return self.get_pred_answer(idx)
        else:
            if "```" in pred_str:
                pred_str = pred_str.split("```")[1]
            if "while" in pred_str:
                return -1
            g = {}
            l = {}
            
            try:
                exec(pred_str.strip(), g, l)
                return l["solver"]()
            except Exception as e:
                return -1
    
    def get_text_answer(self, idx):
        """
        Get the text answer
        
        Args:
            idx: Data index
            
        Returns:
            Text answer
        """
        return self.get_last_pred_text(idx).split("####")[-1]
    
    def judge_correct(self, idx, mode="nl"):
        """
        Judge if a prediction is correct
        
        Args:
            idx: Data index
            mode: Evaluation mode (nl, tool, pot)
            
        Returns:
            True if correct, False otherwise
        """
        golden_answer_str = self.get_origin_input(idx)["answer"].replace(",", "").strip(".").split("\n#### ")[-1]
        try:
            golden_answer = round(float(golden_answer_str), 2)
        except:
            # For non-numerical answers
            golden_answer = golden_answer_str.strip()
            
        if mode == "nl":
            pred = self.get_pred_answer(idx)
        elif mode == "tool":
            pred = self.get_parsed_pred_answer(idx)
        elif mode == "pot":
            pred = self.get_program_answer(idx)
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        
        # Numerical answer comparison
        if isinstance(golden_answer, float) and judge_error(pred):
            return abs(abs(round(float(pred), 2)) - abs(round(golden_answer, 2))) < 0.01
        
        # String answer comparison
        if isinstance(golden_answer, str) and isinstance(pred, str):
            return golden_answer.lower() == pred.lower()
        
        return False


class MMRequestor:
    """
    Class for making requests to language models
    """
    def __init__(self,
                 model_type="openai",
                 model_name="gpt-4",
                 api_key=None,
                 enable_multi_turn=False,
                 api_base=None) -> None:
        """
        Initialize requestor
        
        Args:
            model_type: Type of model API ("openai", "anthropic", etc.)
            model_name: Name of the model
            api_key: API key
            enable_multi_turn: Whether to enable multi-turn conversations
            api_base: Base URL for API (for alternative endpoints)
        """
        self.model_type = model_type
        self.model_name = model_name
        self.enable_multi_turn = enable_multi_turn
        self.chat = []
        
        if model_type == "openai":
            # Import inside method to make dependency optional
            from openai import AsyncOpenAI
            
            # Load API key from environment if not provided
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
            
            # Create client
            if api_base is not None:
                client = AsyncOpenAI(api_key=api_key, base_url=api_base)
            else:
                client = AsyncOpenAI(api_key=api_key)
            
            self.requestor = client
            
        elif model_type == "anthropic":
            # Import inside method to make dependency optional
            import anthropic
            
            # Load API key from environment if not provided
            if api_key is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if api_key is None:
                    raise ValueError("Anthropic API key must be provided or set as ANTHROPIC_API_KEY environment variable")
            
            # Create client
            client = anthropic.AsyncAnthropic(api_key=api_key)
            self.requestor = client
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    async def request(self, prompts, **kwargs):
        """
        Make a request to the language model
        
        Args:
            prompts: Prompt text (string or list of strings)
            **kwargs: Additional arguments for the API
            
        Returns:
            Model response(s)
        """
        if self.model_type == "openai":
            if isinstance(prompts, list):
                responses = []
                for prompt in prompts:
                    self.chat.append({
                        "role": "user",
                        "content": [{
                                "type": "text",
                                "text": prompt,
                            }],
                    })
                    response = await self.requestor.chat.completions.create(
                        model=self.model_name,
                        messages=self.chat,
                        **kwargs
                        )
                    self.chat.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": response.choices[0].message.content,
                        }]
                    })
                    responses.append(response)
                
                res_str = deepcopy(self.chat)
                if not self.enable_multi_turn:
                    self.chat = []
                return res_str
            else:
                prompt = prompts
                self.chat.append({
                    "role": "user",
                    "content": [{
                            "type": "text",
                            "text": prompt,
                        }],
                })
                response = await self.requestor.chat.completions.create(
                    model=self.model_name,
                    messages=self.chat,
                    **kwargs
                    )
                self.chat.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": response.choices[0].message.content,
                    }]
                })
                res_str = deepcopy(self.chat)
                if not self.enable_multi_turn:
                    self.chat = []
                return res_str
        
        elif self.model_type == "anthropic":
            if isinstance(prompts, list):
                responses = []
                for prompt in prompts:
                    if not self.chat:
                        # First message
                        response = await self.requestor.messages.create(
                            model=self.model_name,
                            max_tokens=kwargs.get("max_tokens", 1024),
                            temperature=kwargs.get("temperature", 0.7),
                            system=kwargs.get("system_prompt", ""),
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        self.chat = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response.content[0].text}
                        ]
                    else:
                        # Subsequent messages
                        messages = []
                        for msg in self.chat:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                        messages.append({"role": "user", "content": prompt})
                        
                        response = await self.requestor.messages.create(
                            model=self.model_name,
                            max_tokens=kwargs.get("max_tokens", 1024),
                            temperature=kwargs.get("temperature", 0.7),
                            system=kwargs.get("system_prompt", ""),
                            messages=messages
                        )
                        self.chat.append({"role": "user", "content": prompt})
                        self.chat.append({"role": "assistant", "content": response.content[0].text})
                    
                    # Convert to standard format
                    formatted_chat = []
                    for msg in self.chat:
                        formatted_chat.append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"]}]
                        })
                    responses.append(formatted_chat)
                
                res_str = formatted_chat
                if not self.enable_multi_turn:
                    self.chat = []
                return res_str
            else:
                # Single prompt
                prompt = prompts
                
                if not self.chat:
                    # First message
                    response = await self.requestor.messages.create(
                        model=self.model_name,
                        max_tokens=kwargs.get("max_tokens", 1024),
                        temperature=kwargs.get("temperature", 0.7),
                        system=kwargs.get("system_prompt", ""),
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    self.chat = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response.content[0].text}
                    ]
                else:
                    # Subsequent messages
                    messages = []
                    for msg in self.chat:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    messages.append({"role": "user", "content": prompt})
                    
                    response = await self.requestor.messages.create(
                        model=self.model_name,
                        max_tokens=kwargs.get("max_tokens", 1024),
                        temperature=kwargs.get("temperature", 0.7),
                        system=kwargs.get("system_prompt", ""),
                        messages=messages
                    )
                    self.chat.append({"role": "user", "content": prompt})
                    self.chat.append({"role": "assistant", "content": response.content[0].text})
                
                # Convert to standard format
                formatted_chat = []
                for msg in self.chat:
                    formatted_chat.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                
                res_str = formatted_chat
                if not self.enable_multi_turn:
                    self.chat = []
                return res_str
