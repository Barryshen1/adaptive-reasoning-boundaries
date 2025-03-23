"""
Implementation of Dynamic Boundary Estimation (DBE)
"""
import re
import numpy as np
from collections import defaultdict


class DBE:
    """
    Dynamic Boundary Estimation (DBE)
    
    DBE enables real-time assessment of model reasoning boundaries during interaction 
    through a series of calibrated probes and adaptive prompting adjustments.
    """
    
    def __init__(self, 
                 gamma=0.12,               # Error sensitivity parameter
                 probe_frequency=5,         # How often to deploy probes
                 probe_set_size=7,          # Number of probes per assessment
                 initial_boundary_estimates=None  # Initial estimates if available
                 ):
        """
        Initialize DBE with parameter settings
        
        Args:
            gamma: Error sensitivity parameter
            probe_frequency: How often to deploy probes (every N interactions)
            probe_set_size: Number of probes to use per assessment
            initial_boundary_estimates: Initial boundary estimates if available
        """
        self.gamma = gamma
        self.probe_frequency = probe_frequency
        self.probe_set_size = probe_set_size
        self.boundary_estimates = initial_boundary_estimates or {}
        self.interaction_history = []
        self.error_patterns = defaultdict(list)
        self.confidence_metrics = {}
        self.probes = self._initialize_probes()
    
    def _initialize_probes(self):
        """
        Initialize probes for different reasoning dimensions
        
        Returns:
            Dictionary of probes by category
        """
        return {
            "calculation": self._get_calculation_probes(),
            "planning": self._get_planning_probes(),
            "working_memory": self._get_working_memory_probes()
        }
    
    def _get_calculation_probes(self):
        """
        Generate calculation probes of increasing complexity
        
        Returns:
            List of calculation probes
        """
        return [
            {"query": "What is 58392 × 7201?", "difficulty": 1.0},
            {"query": "Calculate 27531 × 9874 + 57890.", "difficulty": 1.5},
            {"query": "If 823.47 × y = 1.925 × 10^6, what is y?", "difficulty": 2.0},
            {"query": "Calculate (3.14159 × 10^4) ÷ (2.71828 × 10^-2).", "difficulty": 2.5},
            {"query": "Compute the value of 2^18 × 3^12 ÷ 6^9.", "difficulty": 3.0},
            {"query": "Calculate the value of Σ(i=1 to 100) i^2.", "difficulty": 3.5},
            {"query": "Find the value of ∫(0 to 1) x^3 e^x dx.", "difficulty": 4.0},
        ]
    
    def _get_planning_probes(self):
        """
        Generate planning probes requiring different numbers of steps
        
        Returns:
            List of planning probes
        """
        return [
            {"query": "A train travels at 60 km/h. How far does it travel in 2.5 hours?", "difficulty": 1.0},
            {"query": "A store offers a 20% discount, then takes an additional 15% off the discounted price. What is the total percentage discount?", "difficulty": 2.0},
            {"query": "A rectangular pool is 15m long and 10m wide. If you walk around the perimeter 4 times, how many meters will you have walked?", "difficulty": 1.5},
            {"query": "If it takes 6 machines 12 days to produce 480 widgets, how many days would it take 8 machines to produce 960 widgets?", "difficulty": 2.5},
            {"query": "In a tournament, each team plays against every other team exactly once. If there are 10 teams, how many matches will be played?", "difficulty": 2.0},
            {"query": "A mixture of 40 liters contains water and alcohol in the ratio 3:1. How much alcohol must be added to make the ratio 3:2?", "difficulty": 3.0},
            {"query": "A container has 8 red balls, 6 blue balls, and 4 green balls. If 3 balls are drawn randomly without replacement, what is the probability of getting exactly 2 red balls and 1 blue ball?", "difficulty": 3.5},
        ]
    
    def _get_working_memory_probes(self):
        """
        Generate working memory probes testing retention across steps
        
        Returns:
            List of working memory probes
        """
        return [
            {"query": "If a = 5, b = 7, and c = 9, what is the value of a + b + c?", "difficulty": 1.0},
            {"query": "If x = 3, y = 2x, z = y + 4, and w = z^2, what is the value of w?", "difficulty": 2.0},
            {"query": "Let u = 10, v = u/2, w = v^2, x = w - 5, y = 3x, and z = y + 2. What is z?", "difficulty": 2.5},
            {"query": "Define a sequence where a_1 = 3, a_2 = 5, and a_n = a_{n-1} + a_{n-2} for n ≥ 3. What is the value of a_6?", "difficulty": 3.0},
            {"query": "If p = 2, q = p^3, r = q - p, s = r/p, t = s + q, and u = t^2 - r, what is u?", "difficulty": 3.5},
            {"query": "Let sequence a be defined by a_1 = 2, a_2 = 5, and a_n = a_{n-1} × a_{n-2} for n ≥ 3. What is the value of a_5?", "difficulty": 3.5},
            {"query": "Define f(x) = x^2 + 1, g(x) = 2x - 3, and h(x) = f(g(x)) - g(f(x)). What is h(4)?", "difficulty": 4.0},
        ]
    
    def deploy_probes(self, model, probe_category=None, num_probes=None):
        """
        Deploy probes to assess model capabilities
        
        Args:
            model: The language model to probe
            probe_category: Specific category to probe (if None, probe all)
            num_probes: Number of probes to deploy (defaults to self.probe_set_size)
            
        Returns:
            Probe results and selected probes
        """
        # In real implementation, this would use the actual model API
        # For this template, we'll simulate responses
        
        num_probes = num_probes or self.probe_set_size
        selected_probes = []
        
        if probe_category:
            # Select from specific category
            category_probes = self.probes[probe_category]
            selected_indices = np.random.choice(len(category_probes), 
                                               min(num_probes, len(category_probes)), 
                                               replace=False)
            selected_probes = [category_probes[i] for i in selected_indices]
        else:
            # Select across categories
            categories = list(self.probes.keys())
            probes_per_category = max(1, num_probes // len(categories))
            
            for category in categories:
                category_probes = self.probes[category]
                selected_indices = np.random.choice(len(category_probes), 
                                                  min(probes_per_category, len(category_probes)), 
                                                  replace=False)
                selected_probes.extend([category_probes[i] for i in selected_indices])
        
        # Simulate responses (in real implementation, call the model API)
        results = []
        for probe in selected_probes:
            # Simulated response - would be replaced with actual model call
            simulated_correct = np.random.random() > (probe["difficulty"] / 5.0)
            results.append({
                "probe": probe,
                "correct": simulated_correct,
                # Additional metrics that would be extracted from actual responses
                "response_time": np.random.normal(1.0 + probe["difficulty"] * 0.5, 0.2),
                "confidence_markers": np.random.normal(1.0 - probe["difficulty"] * 0.2, 0.1),
                "self_corrections": int(np.random.poisson(probe["difficulty"] - 1.0) if probe["difficulty"] > 1.0 else 0),
                "response": self._generate_simulated_response(probe, simulated_correct)
            })
        
        return results, selected_probes
    
    def _generate_simulated_response(self, probe, correct):
        """
        Generate a simulated model response for a probe
        
        Args:
            probe: The probe
            correct: Whether the answer is correct
            
        Returns:
            Simulated response text
        """
        difficulty = probe["difficulty"]
        query = probe["query"]
        
        # Add hesitation markers based on difficulty
        hesitation_markers = [
            "I think", "Let me calculate this", "This seems like", 
            "If I remember correctly", "I believe", "Probably"
        ]
        
        # Add self-correction markers based on difficulty
        correction_markers = [
            "Wait, I made a mistake", "Let me recalculate", "Actually", 
            "Sorry, I meant", "Correction:"
        ]
        
        # Generate simulated response
        if correct:
            if difficulty < 2.0:
                # Low difficulty, confident response
                return f"The answer is {np.random.randint(100, 1000)}."
            elif difficulty < 3.0:
                # Medium difficulty, some hesitation
                marker = np.random.choice(hesitation_markers) if np.random.random() < 0.5 else ""
                return f"{marker} The answer is {np.random.randint(100, 1000)}."
            else:
                # High difficulty, potential self-correction
                if np.random.random() < 0.3:
                    correction = np.random.choice(correction_markers)
                    return f"The answer is {np.random.randint(100, 1000)}. {correction} The answer is actually {np.random.randint(1000, 5000)}."
                else:
                    marker = np.random.choice(hesitation_markers)
                    return f"{marker} After working through this step by step, I think the answer is {np.random.randint(1000, 5000)}."
        else:
            # Incorrect response with hesitation or confusion
            marker = np.random.choice(hesitation_markers)
            if np.random.random() < 0.3:
                return f"{marker} I'm not entirely sure, but I think the answer is {np.random.randint(100, 1000)}?"
            else:
                return f"{marker} The answer is {np.random.randint(100, 1000)}."
    
    def estimate_boundaries(self, probe_results, model):
        """
        Estimate boundaries based on probe responses
        
        Args:
            probe_results: Results from deployed probes
            model: The language model
            
        Returns:
            Updated boundary estimates
        """
        category_results = defaultdict(list)
        
        # Group results by category
        for result in probe_results:
            probe_category = self._identify_probe_category(result["probe"])
            category_results[probe_category].append(result)
        
        # Estimate boundary for each category
        for category, results in category_results.items():
            # Sort by difficulty
            sorted_results = sorted(results, key=lambda x: x["probe"]["difficulty"])
            
            # Find boundary (highest difficulty with correct answer)
            boundary = 0
            for result in sorted_results:
                if result["correct"]:
                    boundary = max(boundary, result["probe"]["difficulty"])
                else:
                    # If incorrect but simpler problems were correct, this might be the boundary
                    if boundary > 0 and result["probe"]["difficulty"] > boundary:
                        # We'll set boundary between last correct and first incorrect
                        boundary = (boundary + result["probe"]["difficulty"]) / 2
                        break
            
            # Update boundary estimate
            if category not in self.boundary_estimates:
                self.boundary_estimates[category] = boundary
            else:
                # Exponential moving average to update estimate
                self.boundary_estimates[category] = 0.7 * self.boundary_estimates[category] + 0.3 * boundary
        
        return self.boundary_estimates
    
    def _identify_probe_category(self, probe):
        """
        Identify the category a probe belongs to
        
        Args:
            probe: The probe to categorize
            
        Returns:
            Category name
        """
        # In a real implementation, this would look at the probe content
        # For this template, we'll check which category contains this probe
        
        for category, probes in self.probes.items():
            if any(p["query"] == probe["query"] for p in probes):
                return category
        
        # Default fallback
        return "unknown"
    
    def analyze_error_patterns(self, interaction_history):
        """
        Analyze error patterns to refine boundary estimates
        
        Args:
            interaction_history: History of model interactions
            
        Returns:
            Error pattern analysis
        """
        # This would analyze actual model responses in a real implementation
        # For this template, we'll provide a simulated analysis
        
        error_categories = {
            "calculation": {
                "frequency": 0.0,
                "severity": 0.0
            },
            "planning": {
                "frequency": 0.0,
                "severity": 0.0
            },
            "working_memory": {
                "frequency": 0.0,
                "severity": 0.0
            }
        }
        
        # In a real implementation, we would analyze the interaction history
        # and extract patterns of errors to update these metrics
        
        # Simulated analysis
        for category in error_categories:
            error_categories[category]["frequency"] = np.random.beta(2, 5)
            error_categories[category]["severity"] = np.random.beta(2, 5)
        
        return error_categories
    
    def refine_boundary_estimates(self, error_patterns):
        """
        Refine boundary estimates based on error pattern analysis
        
        Args:
            error_patterns: Analyzed error patterns
            
        Returns:
            Refined boundary estimates
        """
        refined_estimates = self.boundary_estimates.copy()
        
        for category, boundary in self.boundary_estimates.items():
            if category in error_patterns:
                error_factor = error_patterns[category]["frequency"] * error_patterns[category]["severity"]
                refined_estimates[category] = boundary * (1 - self.gamma * error_factor)
        
        return refined_estimates
    
    def calibrate_confidence(self, probe_results):
        """
        Calibrate confidence based on response patterns
        
        Args:
            probe_results: Results from deployed probes
            
        Returns:
            Confidence calibration factors
        """
        confidence_factors = {}
        
        for result in probe_results:
            category = self._identify_probe_category(result["probe"])
            
            if category not in confidence_factors:
                confidence_factors[category] = []
            
            # Extract confidence metrics from response
            response_text = result.get("response", "")
            
            # Hesitation markers (e.g., "I think", "probably", "might")
            hesitation_pattern = r'\b(think|probably|might|possibly|perhaps|maybe|not sure|uncertain)\b'
            hesitation = len(re.findall(hesitation_pattern, response_text.lower())) * 0.1
            hesitation = min(1.0, hesitation + 0.2 if "?" in response_text else hesitation)
            
            # Self-correction patterns
            correction_pattern = r'\b(actually|correction|mistake|error|oops|let me recalculate|no wait|sorry)\b'
            corrections = len(re.findall(correction_pattern, response_text.lower())) * 0.2
            
            # Response time factor (longer time suggests lower confidence)
            response_time_factor = min(1.0, result.get("response_time", 1.0) * 0.2)
            
            # Consistency in repeated calculations (simulated)
            consistency = 0.0
            if "repeated_calculations" in result:
                unique_answers = set([calc["answer"] for calc in result.get("repeated_calculations", [])])
                consistency = 0.3 if len(unique_answers) > 1 else 0.0
            
            # Overall confidence metric (lower is less confident)
            confidence = 1.0 - min(1.0, (hesitation + corrections + response_time_factor + consistency))
            
            confidence_factors[category].append(confidence)
        
        # Aggregate confidence factors
        for category, factors in confidence_factors.items():
            if factors:
                confidence_factors[category] = sum(factors) / len(factors)
            else:
                confidence_factors[category] = 1.0  # Default if no data
        
        return confidence_factors
    
    def calibrate_boundary_estimates(self, refined_estimates, confidence_factors):
        """
        Apply confidence calibration to boundary estimates
        
        Args:
            refined_estimates: Refined boundary estimates
            confidence_factors: Confidence calibration factors
            
        Returns:
            Calibrated boundary estimates
        """
        calibrated_estimates = refined_estimates.copy()
        
        for category, boundary in refined_estimates.items():
            if category in confidence_factors:
                # Scale boundary by confidence factor
                # Lower confidence means more conservative boundary estimate
                calibrated_estimates[category] = boundary * confidence_factors[category]
        
        return calibrated_estimates
    
    def compute_combined_boundary(self, calibrated_estimates):
        """
        Compute combined boundary using the combination law
        
        Args:
            calibrated_estimates: Calibrated boundary estimates
            
        Returns:
            Combined boundary estimate
        """
        # Implementation of the combination law from the paper
        # B_{Acc=K}(t_1, t_2, ..., t_n|m) ≈ 1 / ((n-1) * sum(N_i / (B_{Acc=K}(t_i|m) - b_i)))
        
        if not calibrated_estimates:
            return 0.0
        
        # Set default scaling factors
        N_values = {
            "calculation": 1.6e5,
            "planning": 7.0,
            "working_memory": 100.0
        }
        
        b_values = {
            "calculation": 20000,
            "planning": 0.0,
            "working_memory": 1.0
        }
        
        # Calculate the sum term
        sum_term = 0.0
        valid_categories = 0
        
        for category, boundary in calibrated_estimates.items():
            if category in N_values and category in b_values:
                N_i = N_values[category]
                b_i = b_values[category]
                
                # Avoid division by zero or negative values
                denominator = max(0.001, boundary - b_i)
                sum_term += N_i / denominator
                valid_categories += 1
        
        # Calculate combined boundary
        if valid_categories > 1 and sum_term > 0:
            return 1.0 / ((valid_categories - 1) * sum_term)
        elif valid_categories == 1:
            # With only one category, return its boundary
            return list(calibrated_estimates.values())[0]
        else:
            return 0.0
    
    def adapt_prompting_strategy(self, task, combined_boundary, previous_prompt=None, previous_response=None):
        """
        Generate adapted prompting strategy based on boundary estimates
        
        Args:
            task: The reasoning task
            combined_boundary: Combined boundary estimate
            previous_prompt: Previous prompt (if any)
            previous_response: Previous model response (if any)
            
        Returns:
            Adapted prompting strategy
        """
        # This would be a more sophisticated implementation in practice
        # For this template, we'll provide a basic adaptive strategy
        
        # Determine task difficulty (this would be more sophisticated in practice)
        task_difficulty = self._estimate_task_difficulty(task)
        
        if task_difficulty <= combined_boundary * 0.8:
            # Task within comfortable boundary - standard CoT
            prompt = f"Please solve this step-by-step:\n{task}"
        elif task_difficulty <= combined_boundary * 1.2:
            # Task near boundary - structured decomposition
            prompt = f"Break this problem into smaller steps and solve each step carefully:\n{task}"
        else:
            # Task beyond boundary - aggressive scaffolding
            prompt = self._generate_scaffolded_prompt(task, combined_boundary)
        
        # If we have previous response, analyze it for further adaptation
        if previous_prompt and previous_response:
            # In a real implementation, analyze the response for errors or confusion
            # and adapt the prompt accordingly
            if "I'm not sure" in previous_response or "I don't know" in previous_response:
                prompt = self._generate_scaffolded_prompt(task, combined_boundary * 0.7)
            elif any(marker in previous_response for marker in ["actually", "correction", "mistake"]):
                prompt = f"Let's approach this problem step by step, making sure each calculation is precise:\n{task}"
        
        return prompt
    
    def _estimate_task_difficulty(self, task):
        """
        Estimate the difficulty of a task
        
        Args:
            task: The reasoning task
            
        Returns:
            Estimated difficulty
        """
        # Count words as basic complexity measure
        word_count = len(task.split())
        
        # Count special keywords that indicate complexity
        complexity_indicators = [
            "complex", "challenging", "difficult",
            "multi-step", "advanced", "calculate",
            "prove", "explain", "analyze"
        ]
        
        difficulty_score = 1.0  # Base difficulty
        
        # Adjust based on presence of indicators
        for indicator in complexity_indicators:
            if indicator in task.lower():
                difficulty_score += 0.5
        
        # Adjust based on length (longer tasks are often more complex)
        difficulty_score += len(task.split()) / 100
        
        # Count numbers as indicators of calculation complexity
        number_count = len(re.findall(r'\d+', task))
        difficulty_score += number_count * 0.1
        
        # Check for mathematical operations
        if any(op in task for op in ["+", "-", "*", "/", "^", "="]):
            difficulty_score += 0.5
        
        return difficulty_score
    
    def _generate_scaffolded_prompt(self, task, boundary):
        """
        Generate a heavily scaffolded prompt for difficult tasks
        
        Args:
            task: The reasoning task
            boundary: Boundary estimate
            
        Returns:
            Scaffolded prompt
        """
        prompt = f"I'll help you break down this complex problem into very small, manageable steps.\n\n"
        prompt += f"Task: {task}\n\n"
        prompt += "Follow these specific steps:\n\n"
        
        # Generate scaffolding based on task type
        if any(keyword in task.lower() for keyword in ["calculate", "compute", "find", "solve"]):
            # Mathematical scaffolding
            prompt += "1. Identify the key variables and what you're solving for.\n"
            prompt += "2. Write down any relevant formulas or equations.\n"
            prompt += "3. Substitute the known values into these equations.\n"
            prompt += "4. Perform one calculation at a time, showing each step.\n"
            prompt += "5. Double-check your arithmetic at each step.\n"
            prompt += "6. State your final answer clearly.\n\n"
        else:
            # General reasoning scaffolding
            prompt += "1. Identify the key information and constraints.\n"
            prompt += "2. Break down the main question into smaller questions.\n"
            prompt += "3. Address each sub-question one by one.\n"
            prompt += "4. Connect your findings to form a coherent answer.\n"
            prompt += "5. Verify that your answer addresses the original question.\n\n"
        
        prompt += "Now, let's solve this step by step:"
        
        return prompt
    
    def process_interaction(self, task, model, current_interaction=0):
        """
        Process a single interaction in the DBE framework
        
        Args:
            task: The reasoning task
            model: The language model
            current_interaction: Current interaction number
            
        Returns:
            Adapted prompt and updated boundary estimates
        """
        # Determine if we need to deploy probes
        deploy_probes_now = (current_interaction % self.probe_frequency == 0 or 
                            not self.boundary_estimates)
        
        if deploy_probes_now:
            # Deploy probes to assess boundaries
            probe_results, _ = self.deploy_probes(model)
            
            # Update boundary estimates
            self.boundary_estimates = self.estimate_boundaries(probe_results, model)
            
            # Analyze error patterns
            error_patterns = self.analyze_error_patterns(self.interaction_history)
            
            # Refine boundary estimates
            refined_estimates = self.refine_boundary_estimates(error_patterns)
            
            # Calibrate confidence
            confidence_factors = self.calibrate_confidence(probe_results)
            
            # Apply confidence calibration
            calibrated_estimates = self.calibrate_boundary_estimates(refined_estimates, confidence_factors)
            
            # Store updated estimates
            self.boundary_estimates = calibrated_estimates
        
        # Compute combined boundary
        combined_boundary = self.compute_combined_boundary(self.boundary_estimates)
        
        # Get previous prompt and response if available
        previous_prompt = None
        previous_response = None
        if self.interaction_history:
            previous = self.interaction_history[-1]
            previous_prompt = previous.get("prompt")
            previous_response = previous.get("response")
        
        # Generate adapted prompt
        adapted_prompt = self.adapt_prompting_strategy(task, combined_boundary, 
                                                     previous_prompt, previous_response)
        
        # Update interaction history (in real implementation, would include the response)
        self.interaction_history.append({
            "prompt": adapted_prompt,
            "response": None,  # Would be filled with actual response
            "boundary_estimates": self.boundary_estimates.copy(),
            "combined_boundary": combined_boundary
        })
        
        return adapted_prompt, self.boundary_estimates, combined_boundary
