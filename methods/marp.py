"""
Implementation of Minimum Acceptable Reasoning Paths (MARP)
"""
import re
import numpy as np
from utils.request_tool import MMRequestor


class MARP:
    """
    Minimum Acceptable Reasoning Paths (MARP)
    
    MARP optimizes reasoning within model boundaries by:
    1. Setting an upper limit on single-step computational capacity
    2. Maximizing computation per step while reducing the number of global planning steps
    """
    
    def __init__(self, 
                 max_operations_per_step=5,  # Maximum operations per step
                 max_multiplication_value=1.5e5,  # Maximum multiplication value
                 api_key=None,
                 verbose=False):
        """
        Initialize MARP with parameter settings
        
        Args:
            max_operations_per_step: Maximum operations allowed per step
            max_multiplication_value: Maximum value for multiplication operations
            api_key: API key for model access
            verbose: Whether to print detailed information
        """
        self.max_operations_per_step = max_operations_per_step
        self.max_multiplication_value = max_multiplication_value
        self.api_key = api_key
        self.verbose = verbose
    
    def generate_prompt(self, task):
        """
        Generate MARP prompt for a task
        
        Args:
            task: The reasoning task
            
        Returns:
            MARP prompt
        """
        instruction = f"""You need to perform multi-step reasoning, with each step carrying out as many basic operations as possible.

Remember, you can only complete tasks that contain up to {self.max_operations_per_step} basic operations per step, and multiplication operations must be less than {self.max_multiplication_value}. The upper limit of the multiplication operations decreases as the number of operations per step increases.

[EXAMPLE]
Question: Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?
Answer: Leo finished the first and second parts of the assignment in 25 + 25*2 = <<25+25*2=75>>75 minutes.
Therefore, it took Leo 60 x 2 - 75 = <<60*2-75=45>>45 minutes to finish the third part of the assignment.
#### 45

Question: Liza bought 10 kilograms of butter to make cookies. She used one-half of it for chocolate chip cookies, one-fifth of it for peanut butter cookies, and one-third of the remaining butter for sugar cookies. How many kilograms of butter are left after making those three kinds of cookies?
Answer: Liza used 10 / 2 + 10 / 5 = <<10/2+10/5=7>>7 kilograms of butter for the chocolate and peanut butter cookies.
Then, Liza used (10 - 7) / 3 = <<(10-7)/3=1>>1 kilograms of butter for the sugar cookies.
Therefore, only 10-7-1 = <<10-7-1=2>>2 kilograms of butter were left.
#### 2

Question: Tina makes $18 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
Answer: She works 5 days and makes 5 * 8 * $18 = $<<8*18*5=720>>720 regular pay.
Her overtime pay is 18+18*0.5 = $<<18+18*0.5=27>>27.
She works 2 hours of overtime for 5 days and makes 27*2*5 = $<<27*(10-8)*5=270>>270 in overtime pay.
She makes $720 + $270 = $<<720+270=990>>990.
#### 990

[REQUEST]
Question: {task}"""
        
        return instruction
    
    async def process_task(self, task, requestor=None, model_name=None, api_key=None):
        """
        Process a reasoning task with MARP
        
        Args:
            task: The reasoning task
            requestor: Optional requestor for API calls
            model_name: Optional model name
            api_key: Optional API key
            
        Returns:
            Task result
        """
        # Generate the MARP prompt
        prompt = self.generate_prompt(task)
        
        # Create requestor if not provided
        if requestor is None:
            if api_key is None:
                api_key = self.api_key
                
            if model_name is None:
                model_name = "gpt-3.5-turbo"  # Default model
                
            model_type = "openai" if "gpt" in model_name.lower() else "anthropic"
            requestor = MMRequestor(
                model_type=model_type,
                model_name=model_name,
                api_key=api_key
            )
        
        # Make the API request
        try:
            response = await requestor.request(prompt, temperature=0.2, max_tokens=300)
            
            # Extract the response text
            if isinstance(response, list):
                response_text = response[-1]["content"][0]["text"]
            else:
                response_text = response
                
            if self.verbose:
                print(f"Response: {response_text[:100]}...")
                
            return {
                "prompt": prompt,
                "response": response_text,
                "method": "marp"
            }
        except Exception as e:
            if self.verbose:
                print(f"Error processing task: {e}")
            return {
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "method": "marp"
            }
