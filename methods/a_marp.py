"""
Implementation of Advanced Minimum Acceptable Reasoning Paths (A-MARP)
"""
import re
import numpy as np
from methods.dbe import DBE  # Import DBE directly


class AMARP:
    """
    Advanced Minimum Acceptable Reasoning Paths (A-MARP)
    
    A-MARP leverages dynamic boundary estimates from DBE with four key enhancements:
    1. Adaptive Step Calibration
    2. Difficulty-Aware Decomposition
    3. Contextual Boundary Awareness
    4. Memory-Augmented Reasoning
    """
    
    def __init__(self, 
                 alpha=0.15,  # Step calibration parameter
                 beta=0.08,   # Context sensitivity parameter
                 c_max=5,     # Maximum operations per step
                 dbe_instance=None,  # Reference to DBE instance for boundary estimates
                 dbe_params=None,   # Parameters to initialize DBE if not provided
                 ):
        """
        Initialize A-MARP with parameter settings
        
        Args:
            alpha: Calibration parameter for step complexity
            beta: Context sensitivity parameter
            c_max: Maximum operations per step
            dbe_instance: Reference to a DBE instance for boundary estimates
            dbe_params: Parameters to initialize a new DBE instance if dbe_instance is None
        """
        self.alpha = alpha
        self.beta = beta
        self.c_max = c_max
        
        # Create DBE instance if not provided
        if dbe_instance is None:
            # Default DBE parameters if not provided
            if dbe_params is None:
                dbe_params = {
                    "gamma": 0.12,
                    "probe_frequency": 5,
                    "probe_set_size": 7
                }
            self.dbe_instance = DBE(**dbe_params)
            print("A-MARP initialized with a new DBE instance")
        else:
            self.dbe_instance = dbe_instance
            print("A-MARP initialized with provided DBE instance")
            
        self.memory = {}
    
    def get_boundary_estimates(self, model, task=None):
        """
        Get boundary estimates from DBE
        
        Args:
            model: The language model to get estimates for
            task: The reasoning task (optional)
            
        Returns:
            Dictionary of estimated boundaries
        """
        # Always use DBE for boundary estimates
        if hasattr(self.dbe_instance, 'boundary_estimates') and self.dbe_instance.boundary_estimates:
            return self.dbe_instance.boundary_estimates
        
        # If DBE has calibrated boundaries, use those
        if hasattr(self.dbe_instance, 'get_calibrated_boundaries'):
            return self.dbe_instance.get_calibrated_boundaries()
        
        # If no boundary estimates available yet, request a probe deployment
        # This would typically happen asynchronously with the model
        print("Warning: No DBE boundary estimates available. DBE probes should be deployed first.")
        return {
            "calculation": None,
            "planning": None,
            "working_memory": None
        }
    
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
        # Get boundary estimates from DBE
        boundary_estimates = self.get_boundary_estimates(model, task)
        
        # Extract calculation boundary
        if boundary_estimates and 'calculation' in boundary_estimates and boundary_estimates['calculation'] is not None:
            b_c = boundary_estimates['calculation']
        else:
            # Temporary value only if DBE hasn't generated estimates yet
            # This should be replaced by actual DBE estimates as soon as they're available
            print("Warning: Waiting for DBE to generate calculation boundary estimates")
            b_c = 100000  # Temporary placeholder until DBE estimates are ready
        
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
        # Get boundary estimates from DBE
        boundary_estimates = self.get_boundary_estimates(model, task)
        
        # Extract planning boundary
        if boundary_estimates and 'planning' in boundary_estimates and boundary_estimates['planning'] is not None:
            b_p = boundary_estimates['planning']
        else:
            # Temporary value only if DBE hasn't generated estimates yet
            print("Warning: Waiting for DBE to generate planning boundary estimates")
            b_p = 7.0  # Temporary placeholder until DBE estimates are ready
        
        if total_complexity is None:
            # Estimate total complexity if not provided
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
        
        # Get boundary estimates from DBE
        boundary_estimates = self.get_boundary_estimates(model, task)
        adjusted_boundaries = {}
        
        if boundary_estimates:
            # Apply contextual adjustment to each boundary
            for dim, value in boundary_estimates.items():
                if value is not None:  # Ensure we only adjust valid boundaries
                    adjusted_boundaries[dim] = value * (1 + self.beta * context_factor)
        else:
            # No adjustments if no boundary estimates available
            print("Warning: Cannot adjust boundaries without DBE estimates")
            return boundary_estimates
            
        return adjusted_boundaries
    
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
        Generate optimized prompting strategy using dynamic boundary estimates
        
        Args:
            task: The reasoning task
            difficulty: Estimated task difficulty
            model: The language model
            
        Returns:
            Structured prompt for the reasoning task
        """
        # Get dynamic boundary estimates and apply contextual adjustment
        boundary_estimates = self.get_boundary_estimates(model, task)
        if boundary_estimates:
            adjusted_boundaries = self.contextual_boundary_adjustment(task, model)
        else:
            adjusted_boundaries = None
            
        # Calculate key parameters using the dynamic boundaries
        c_step = self.adaptive_step_calibration(task, difficulty, model)
        n_steps = self.difficulty_aware_decomposition(task, difficulty, model)
        
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
    
    # Add a method to expose DBE's probe deployment
    async def deploy_dbe_probes(self, requestor, probe_category=None, num_probes=None):
        """
        Deploy DBE probes to assess model capabilities
        
        Args:
            requestor: The language model requestor for API calls
            probe_category: Specific category to probe (if None, probe all)
            num_probes: Number of probes to deploy
            
        Returns:
            Probe results and updated boundary estimates
        """
        # Call DBE's deploy_probes method
        probe_results, selected_probes = await self.dbe_instance.deploy_probes(
            requestor, 
            probe_category,
            num_probes
        )
        
        # Extract model name from requestor if available
        model_name = getattr(requestor, "model_name", "unknown_model")
        
        # Update boundary estimates in DBE
        updated_estimates = self.dbe_instance.estimate_boundaries(probe_results, model_name)
        
        return probe_results, updated_estimates
    
    # Add a method to run a complete experiment with DBE integration
    async def process_with_dbe(self, task, difficulty, model, requestor, interaction_count=0):
        """
        Process a task with integrated DBE
        
        Args:
            task: The reasoning task
            difficulty: Estimated task difficulty 
            model: The language model name
            requestor: The language model requestor for API calls
            interaction_count: Current interaction number
            
        Returns:
            Processed results with DBE integration
        """
        # First, update DBE estimates if needed
        if not self.dbe_instance.boundary_estimates or interaction_count % self.dbe_instance.probe_frequency == 0:
            print(f"Deploying DBE probes at interaction {interaction_count}...")
            probe_results, _ = await self.deploy_dbe_probes(requestor)
        
        # Generate the A-MARP prompt using DBE boundary estimates
        prompt = self.generate_prompt(task, difficulty, model)
        
        # Return the prompt and current boundary estimates for execution
        return {
            "prompt": prompt,
            "boundary_estimates": self.dbe_instance.boundary_estimates,
            "interaction": interaction_count
        }
