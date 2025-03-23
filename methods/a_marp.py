"""
Implementation of Advanced Minimum Acceptable Reasoning Paths (A-MARP)
"""
import re
import numpy as np


class AMARP:
    """
    Advanced Minimum Acceptable Reasoning Paths (A-MARP)
    
    A-MARP extends the original MARP approach with four key enhancements:
    1. Adaptive Step Calibration
    2. Difficulty-Aware Decomposition
    3. Contextual Boundary Awareness
    4. Memory-Augmented Reasoning
    """
    
    def __init__(self, 
                 alpha=0.15,  # Step calibration parameter
                 beta=0.08,   # Context sensitivity parameter
                 c_max=5,     # Maximum operations per step
                 b_c=None,    # Calculation boundary (if known)
                 b_p=None,    # Planning boundary (if known)
                 ):
        """
        Initialize A-MARP with parameter settings
        
        Args:
            alpha: Calibration parameter for step complexity
            beta: Context sensitivity parameter
            c_max: Maximum operations per step
            b_c: Calculation reasoning boundary (if known)
            b_p: Planning reasoning boundary (if known)
        """
        self.alpha = alpha
        self.beta = beta
        self.c_max = c_max
        self.b_c = b_c
        self.b_p = b_p
        self.memory = {}
    
    def estimate_calculation_boundary(self, model):
        """
        Estimate calculation boundary for a given model
        
        Args:
            model: The language model to estimate boundary for
            
        Returns:
            Estimated calculation boundary value
        """
        # Placeholder for actual implementation
        # In practice, this would involve probing the model with calculation tasks
        if self.b_c is not None:
            return self.b_c
        
        # Default values based on paper findings
        boundary_values = {
            'gpt-3.5-turbo': 220000,
            'gpt-4': 1000000,
            'claude-3-opus': 800000,
            'claude-3.7-sonnet': 900000,
            'llama-3-70b': 150000,
            'o1-preview': 2000000,
            'gpt-4.5-orion': 2500000,
            'gpt-o3': 1800000,
            'gpt-o1': 1600000,
        }
        
        return boundary_values.get(model, 220000)
    
    def estimate_planning_boundary(self, model):
        """
        Estimate planning boundary for a given model
        
        Args:
            model: The language model to estimate boundary for
            
        Returns:
            Estimated planning boundary value
        """
        # Placeholder for actual implementation
        if self.b_p is not None:
            return self.b_p
        
        # Default values based on paper findings
        boundary_values = {
            'gpt-3.5-turbo': 7.0,
            'gpt-4': 12.0,
            'claude-3-opus': 10.0,
            'claude-3.7-sonnet': 11.0,
            'llama-3-70b': 5.0,
            'o1-preview': 15.0,
            'gpt-4.5-orion': 18.0,
            'gpt-o3': 14.0,
            'gpt-o1': 13.0,
        }
        
        return boundary_values.get(model, 7.0)
    
    def adaptive_step_calibration(self, task, difficulty, model):
        """
        Dynamically calibrate the optimal step complexity based on task characteristics
        
        Args:
            task: The reasoning task
            difficulty: Estimated task difficulty
            model: The language model
            
        Returns:
            Recommended computational complexity per step
        """
        b_c = self.estimate_calculation_boundary(model)
        return min(b_c / (1 + self.alpha * difficulty), self.c_max)
    
    def difficulty_aware_decomposition(self, task, difficulty, model, total_complexity=None):
        """
        Implement adaptive decomposition based on estimated task difficulty
        
        Args:
            task: The reasoning task
            difficulty: Estimated task difficulty
            model: The language model
            total_complexity: Total reasoning complexity of the task
            
        Returns:
            Recommended number of reasoning steps
        """
        b_p = self.estimate_planning_boundary(model)
        
        if total_complexity is None:
            # Estimate total complexity if not provided
            # This would typically be more sophisticated in practice
            total_complexity = difficulty * 10
        
        return max(int(np.ceil(difficulty * total_complexity / b_p)), 1)
    
    def contextual_boundary_adjustment(self, task, model):
        """
        Adjust boundaries based on domain-specific contextual factors
        
        Args:
            task: The reasoning task
            model: The language model
            
        Returns:
            Adjusted boundary value
        """
        # Identify relevant task category
        task_type = self._identify_task_type(task)
        context_factor = self._calculate_context_factor(task_type)
        
        b_original = self.estimate_calculation_boundary(model)
        return b_original * (1 + self.beta * context_factor)
    
    def _identify_task_type(self, task):
        """
        Identify the type of reasoning task
        
        Args:
            task: The reasoning task
            
        Returns:
            Task type identifier
        """
        # Check for mathematical content
        math_keywords = ["calculate", "solve", "equation", "math", "arithmetic", "compute", 
                         "number", "formula", "sum", "product", "division"]
        if any(keyword in task.lower() for keyword in math_keywords) or re.search(r'\d+[\+\-\*\/]\d+', task):
            return "mathematical"
        
        # Check for logical reasoning content
        logical_keywords = ["logical", "deduce", "infer", "conclude", "premise", "argument", 
                           "valid", "invalid", "syllogism", "if-then"]
        if any(keyword in task.lower() for keyword in logical_keywords):
            return "logical"
        
        # Check for analytical content
        analytical_keywords = ["compare", "contrast", "analyze", "evaluate", "assess", 
                              "critique", "examine", "review", "study"]
        if any(keyword in task.lower() for keyword in analytical_keywords):
            return "analytical"
        
        return "general"
    
    def _calculate_context_factor(self, task_type):
        """
        Calculate context factor for boundary adjustment
        
        Args:
            task_type: Type of reasoning task
            
        Returns:
            Context factor value
        """
        context_factors = {
            "mathematical": 0.2,
            "logical": 0.1,
            "analytical": 0.15,
            "general": 0.0
        }
        
        return context_factors.get(task_type, 0.0)
    
    def update_memory(self, step, result):
        """
        Update memory component with new information
        
        Args:
            step: Current reasoning step
            result: Result of the current step
            
        Returns:
            Updated memory state
        """
        step_id = f"step_{len(self.memory) + 1}"
        
        # Track dependencies between steps
        dependencies = []
        for prev_step_id, prev_step in self.memory.items():
            # Check if current step depends on previous steps
            if any(term in step for term in prev_step.get("result_terms", [])):
                dependencies.append(prev_step_id)
        
        # Extract key terms from the result
        if isinstance(result, str):
            result_terms = list(set(re.findall(r'\b\w+\b', result)))
        else:
            result_terms = []
        
        self.memory[step_id] = {
            "step": step,
            "result": result,
            "result_terms": result_terms,
            "dependencies": dependencies,
            "timestamp": len(self.memory)
        }
        
        return self.memory
    
    def generate_prompt(self, task, difficulty, model):
        """
        Generate optimized prompting strategy
        
        Args:
            task: The reasoning task
            difficulty: Estimated task difficulty
            model: The language model
            
        Returns:
            Structured prompt for the reasoning task
        """
        # Calculate key parameters
        c_step = self.adaptive_step_calibration(task, difficulty, model)
        n_steps = self.difficulty_aware_decomposition(task, difficulty, model)
        adjusted_boundary = self.contextual_boundary_adjustment(task, model)
        
        # Construct the prompt
        prompt = "You need to perform multi-step reasoning with adaptive complexity.\n\n"
        prompt += f"For this task, break your reasoning into approximately {n_steps} steps.\n"
        prompt += f"Each step should contain at most {int(c_step)} operations.\n"
        prompt += "Track important intermediate results in your reasoning.\n\n"
        
        # Add example based on task type
        prompt += self._get_example_prompt(task, n_steps, int(c_step))
        
        # Add the actual task
        prompt += f"\nNow solve the following problem:\n{task}"
        
        return prompt
    
    def _get_example_prompt(self, task, n_steps, c_step):
        """
        Get an example prompt based on task type
        
        Args:
            task: The reasoning task
            n_steps: Recommended number of steps
            c_step: Recommended operations per step
            
        Returns:
            Example prompt
        """
        task_type = self._identify_task_type(task)
        
        if task_type == "mathematical":
            return self._get_math_example(n_steps, c_step)
        elif task_type == "logical":
            return self._get_logical_example(n_steps, c_step)
        else:
            return self._get_general_example(n_steps, c_step)
    
    def _get_math_example(self, n_steps, c_step):
        """
        Generate a mathematical reasoning example
        """
        if n_steps <= 2:
            return """Example:
Question: Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?
Answer: Leo finished the first and second parts of the assignment in 25 + 25*2 = <<25+25*2=75>>75 minutes.
Therefore, it took Leo 60 x 2 - 75 = <<60*2-75=45>>45 minutes to finish the third part of the assignment.
#### 45"""
        else:
            return """Example:
Question: A factory has two production lines. The first line can produce 12 units per hour, and the second can produce 15 units per hour. If the factory runs the first line for 8 hours and the second line for x hours to produce 252 units, find x.
Answer: 
Step 1: Calculate the units produced by the first line in 8 hours.
Units from first line = 12 units/hour × 8 hours = <<12*8=96>>96 units

Step 2: Calculate the remaining units that need to be produced by the second line.
Remaining units = 252 - 96 = <<252-96=156>>156 units

Step 3: Calculate the time needed for the second line to produce the remaining units.
Time for second line = Remaining units ÷ Production rate = 156 ÷ 15 = <<156/15=10.4>>10.4 hours
Therefore, x = 10.4 hours.
#### 10.4"""
    
    def _get_logical_example(self, n_steps, c_step):
        """
        Generate a logical reasoning example
        """
        return """Example:
Question: If all glorks are flurbs, and all flurbs are mips, what can we conclude about glorks?
Answer:
Step 1: Analyze the first statement: "All glorks are flurbs."
This means every glork is also a flurb.

Step 2: Analyze the second statement: "All flurbs are mips."
This means every flurb is also a mip.

Step 3: Apply transitive property to combine the statements.
If every glork is a flurb, and every flurb is a mip, then every glork must also be a mip.

Therefore, we can conclude that all glorks are mips.
#### All glorks are mips."""
    
    def _get_general_example(self, n_steps, c_step):
        """
        Generate a general reasoning example
        """
        return """Example:
Question: What would happen to sea levels if all the ice in Antarctica melted?
Answer:
Step 1: Consider the current state of Antarctic ice.
Antarctica contains about 90% of the world's ice and 70% of its fresh water.

Step 2: Analyze what happens when land ice melts versus sea ice.
When land ice melts, it adds water to the ocean, but when sea ice melts, it doesn't significantly change sea levels since it's already displacing water.

Step 3: Calculate the potential sea level rise.
If all Antarctic land ice melted, global sea levels would rise by approximately 58-60 meters.

Step 4: Consider the implications of such a rise.
This would flood coastal cities worldwide, displace billions of people, and radically alter Earth's geography.

Therefore, if all Antarctic ice melted, sea levels would rise dramatically by about 60 meters, causing catastrophic global flooding.
#### Sea levels would rise by approximately 60 meters."""
