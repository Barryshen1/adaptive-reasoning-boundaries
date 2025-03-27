"""
Implementation of Dynamic Boundary Estimation (DBE)
"""
import re
import numpy as np
import json
import time
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
                 initial_boundary_estimates=None,  # Initial estimates if available
                 results_path="experiments/results/dbe_boundaries.json"  # Path to save boundary estimates
                 ):
        """
        Initialize DBE with parameter settings
        
        Args:
            gamma: Error sensitivity parameter
            probe_frequency: How often to deploy probes (every N interactions)
            probe_set_size: Number of probes to use per assessment
            initial_boundary_estimates: Initial boundary estimates if available
            results_path: Path to save boundary estimates
        """
        self.gamma = gamma
        self.probe_frequency = probe_frequency
        self.probe_set_size = probe_set_size
        self.boundary_estimates = initial_boundary_estimates or {}
        self.results_path = results_path
        self.interaction_history = []
        self.error_patterns = defaultdict(list)
        self.confidence_metrics = {}
        self.probes = self._initialize_probes()
        
        # Load existing boundary estimates if available
        try:
            with open(results_path, 'r') as f:
                saved_boundaries = json.load(f)
                if not self.boundary_estimates and saved_boundaries:
                    self.boundary_estimates = saved_boundaries
                    print(f"Loaded boundary estimates from {results_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"No existing boundary estimates found at {results_path}")
    
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
            {"query": "What is 58392 × 7201?", "difficulty": 1.0, "answer": "420620392"},
            {"query": "Calculate 27531 × 9874 + 57890.", "difficulty": 1.5, "answer": "271836674"},
            {"query": "If 823.47 × y = 1.925 × 10^6, what is y?", "difficulty": 2.0, "answer": "2337.58"},
            {"query": "Calculate (3.14159 × 10^4) ÷ (2.71828 × 10^-2).", "difficulty": 2.5, "answer": "1156104.85"},
            {"query": "Compute the value of 2^18 × 3^12 ÷ 6^9.", "difficulty": 3.0, "answer": "2097152"},
            {"query": "Calculate the value of Σ(i=1 to 100) i^2.", "difficulty": 3.5, "answer": "338350"},
            {"query": "Find the value of ∫(0 to 1) x^3 e^x dx.", "difficulty": 4.0, "answer": "0.462"},
        ]
    
    def _get_planning_probes(self):
        """
        Generate planning probes requiring different numbers of steps
        
        Returns:
            List of planning probes
        """
        return [
            {"query": "A train travels at 60 km/h. How far does it travel in 2.5 hours?", "difficulty": 1.0, "answer": "150 km"},
            {"query": "A store offers a 20% discount, then takes an additional 15% off the discounted price. What is the total percentage discount?", "difficulty": 2.0, "answer": "32%"},
            {"query": "A rectangular pool is 15m long and 10m wide. If you walk around the perimeter 4 times, how many meters will you have walked?", "difficulty": 1.5, "answer": "200 m"},
            {"query": "If it takes 6 machines 12 days to produce 480 widgets, how many days would it take 8 machines to produce 960 widgets?", "difficulty": 2.5, "answer": "18 days"},
            {"query": "In a tournament, each team plays against every other team exactly once. If there are 10 teams, how many matches will be played?", "difficulty": 2.0, "answer": "45 matches"},
            {"query": "A mixture of 40 liters contains water and alcohol in the ratio 3:1. How much alcohol must be added to make the ratio 3:2?", "difficulty": 3.0, "answer": "10 liters"},
            {"query": "A container has 8 red balls, 6 blue balls, and 4 green balls. If 3 balls are drawn randomly without replacement, what is the probability of getting exactly 2 red balls and 1 blue ball?", "difficulty": 3.5, "answer": "0.21"},
        ]
    
    def _get_working_memory_probes(self):
        """
        Generate working memory probes testing retention across steps
        
        Returns:
            List of working memory probes
        """
        return [
            {"query": "If a = 5, b = 7, and c = 9, what is the value of a + b + c?", "difficulty": 1.0, "answer": "21"},
            {"query": "If x = 3, y = 2x, z = y + 4, and w = z^2, what is the value of w?", "difficulty": 2.0, "answer": "100"},
            {"query": "Let u = 10, v = u/2, w = v^2, x = w - 5, y = 3x, and z = y + 2. What is z?", "difficulty": 2.5, "answer": "47"},
            {"query": "Define a sequence where a_1 = 3, a_2 = 5, and a_n = a_{n-1} + a_{n-2} for n ≥ 3. What is the value of a_6?", "difficulty": 3.0, "answer": "29"},
            {"query": "If p = 2, q = p^3, r = q - p, s = r/p, t = s + q, and u = t^2 - r, what is u?", "difficulty": 3.5, "answer": "121"},
            {"query": "Let sequence a be defined by a_1 = 2, a_2 = 5, and a_n = a_{n-1} × a_{n-2} for n ≥ 3. What is the value of a_5?", "difficulty": 3.5, "answer": "500"},
            {"query": "Define f(x) = x^2 + 1, g(x) = 2x - 3, and h(x) = f(g(x)) - g(f(x)). What is h(4)?", "difficulty": 4.0, "answer": "7"},
        ]
    
    async def deploy_probes(self, requestor, probe_category=None, num_probes=None):
        """
        Deploy probes to assess model capabilities
        
        Args:
            requestor: The language model requestor for API calls
            probe_category: Specific category to probe (if None, probe all)
            num_probes: Number of probes to deploy (defaults to self.probe_set_size)
            
        Returns:
            Probe results and selected probes
        """
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
        
        # Issue real API calls to the model
        results = []
        for probe in selected_probes:
            prompt = f"Solve this problem step by step:\n{probe['query']}"
            
            # Time the response to measure latency
            start_time = time.time()
            
            try:
                # Make actual API call
                response = await requestor.request(prompt, temperature=0.7, max_tokens=1024)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Extract the full text of the response
                response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else ""
                
                # Extract answer from the response
                extracted_answer = self._extract_answer(response_text)
                
                # Check if the answer is correct
                is_correct = self._check_answer(extracted_answer, probe["answer"])
                
                # Analyze confidence markers in the response
                confidence_metrics = self._extract_confidence_metrics(response_text)
                
                # Count self-corrections
                self_corrections = self._count_self_corrections(response_text)
                
                results.append({
                    "probe": probe,
                    "correct": is_correct,
                    "response_time": response_time,
                    "confidence_markers": confidence_metrics,
                    "self_corrections": self_corrections,
                    "response": response_text,
                    "extracted_answer": extracted_answer
                })
                
            except Exception as e:
                print(f"Error deploying probe {probe['query']}: {e}")
                # Add a failed result
                results.append({
                    "probe": probe,
                    "correct": False,
                    "error": str(e),
                    "response": None
                })
        
        return results, selected_probes
    
    def _extract_answer(self, response_text):
        """
        Extract numerical answer from model response
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Extracted answer
        """
        if not response_text:
            return None
        
        # Look for answer patterns
        # Pattern 1: After "answer is" or similar phrases
        answer_patterns = [
            r"answer is[:\s]*([+-]?\d+\.?\d*)",
            r"result is[:\s]*([+-]?\d+\.?\d*)",
            r"equals[:\s]*([+-]?\d+\.?\d*)",
            r"value is[:\s]*([+-]?\d+\.?\d*)",
            r"=\s*([+-]?\d+\.?\d*)"
        ]
        
        # Pattern 2: After line markers like "#### " (common in CoT responses)
        if "####" in response_text:
            after_marker = response_text.split("####")[-1].strip()
            numbers = re.findall(r"([+-]?\d+\.?\d*)", after_marker)
            if numbers:
                return numbers[0]
        
        # Try each pattern
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                return matches[-1]
        
        # Pattern 3: Last number in response as fallback
        numbers = re.findall(r"([+-]?\d+\.?\d*)", response_text)
        if numbers:
            return numbers[-1]
        
        return None
    
    def _check_answer(self, extracted_answer, correct_answer):
        """
        Check if the extracted answer is correct
        
        Args:
            extracted_answer: The answer extracted from the response
            correct_answer: The correct answer
            
        Returns:
            True if correct, False otherwise
        """
        if extracted_answer is None:
            return False
        
        try:
            # Convert both to numbers if possible
            extracted_num = float(extracted_answer.replace(",", ""))
            
            # Handle various formats in correct answer
            correct_num = None
            if isinstance(correct_answer, (int, float)):
                correct_num = float(correct_answer)
            elif isinstance(correct_answer, str):
                # Handle units and extract the numerical part
                correct_str = correct_answer.split()[0] if " " in correct_answer else correct_answer
                correct_num = float(correct_str.replace(",", ""))
            
            if correct_num is not None:
                # Allow for small numerical differences
                return abs(extracted_num - correct_num) < max(0.01 * abs(correct_num), 0.001)
        except (ValueError, TypeError):
            # If conversion fails, compare strings (for non-numerical answers)
            return extracted_answer.lower().strip() == correct_answer.lower().strip()
        
        return False
    
    def _extract_confidence_metrics(self, response_text):
        """
        Extract confidence metrics from model response
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Confidence score (0-1)
        """
        if not response_text:
            return 0.5
        
        # Confidence reducers
        hesitation_markers = [
            "i think", "probably", "might", "could be", "possibly", "seems like",
            "not sure", "guess", "assume", "perhaps", "estimate", "approximate"
        ]
        
        # Confidence boosters
        certainty_markers = [
            "definitely", "certainly", "clearly", "exactly", "precisely", "absolutely",
            "without doubt", "evidently", "indeed", "of course", "undoubtedly"
        ]
        
        hesitation_count = sum(response_text.lower().count(marker) for marker in hesitation_markers)
        certainty_count = sum(response_text.lower().count(marker) for marker in certainty_markers)
        
        # Analyze question marks (indicates uncertainty)
        question_marks = response_text.count("?")
        
        # Base confidence score
        base_confidence = 0.7
        
        # Adjust based on markers
        confidence = base_confidence - (hesitation_count * 0.1) + (certainty_count * 0.1) - (question_marks * 0.15)
        
        # Keep in range [0.1, 0.95]
        return max(0.1, min(0.95, confidence))
    
    def _count_self_corrections(self, response_text):
        """
        Count self-corrections in model response
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Number of self-corrections
        """
        if not response_text:
            return 0
        
        correction_markers = [
            "actually", "wait", "correction", "let me recalculate", "i made a mistake", 
            "i need to correct", "let me fix", "let me rethink", "oops", "sorry"
        ]
        
        correction_count = sum(response_text.lower().count(marker) for marker in correction_markers)
        
        # Also check for mathematical corrections (e.g., 24 -> 25)
        math_corrections = len(re.findall(r"\b(\d+\.?\d*)\s*(?:->|→|to)\s*(\d+\.?\d*)\b", response_text))
        
        return correction_count + math_corrections
    
    def estimate_boundaries(self, probe_results, model_name):
        """
        Estimate boundaries based on probe responses
        
        Args:
            probe_results: Results from deployed probes
            model_name: The name of the language model
            
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
        
        # Save updated boundaries to file
        self._save_boundary_estimates(model_name)
        
        return self.boundary_estimates
    
    def _save_boundary_estimates(self, model_name):
        """
        Save boundary estimates to file
        
        Args:
            model_name: Name of the model for which boundaries are estimated
        """
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
            
            # Add model name to the estimates
            data_to_save = {
                "model_name": model_name,
                "boundaries": self.boundary_estimates,
                "timestamp": time.time()
            }
            
            with open(self.results_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
                
            print(f"Saved boundary estimates to {self.results_path}")
        except Exception as e:
            print(f"Error saving boundary estimates: {e}")
    
    def _identify_probe_category(self, probe):
        """
        Identify the category a probe belongs to
        
        Args:
            probe: The probe to categorize
            
        Returns:
            Category name
        """
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
        if not interaction_history:
            return {
                "calculation": {"frequency": 0.0, "severity": 0.0},
                "planning": {"frequency": 0.0, "severity": 0.0},
                "working_memory": {"frequency": 0.0, "severity": 0.0}
            }
        
        error_categories = {
            "calculation": {"frequency": 0.0, "severity": 0.0},
            "planning": {"frequency": 0.0, "severity": 0.0},
            "working_memory": {"frequency": 0.0, "severity": 0.0}
        }
        
        # Count interactions and errors by category
        category_counts = defaultdict(int)
        error_counts = defaultdict(int)
        severity_sum = defaultdict(float)
        
        for interaction in interaction_history:
            # Skip entries without response
            if not interaction.get("response"):
                continue
                
            # Check if there's a prompt and response
            prompt = interaction.get("prompt", "")
            response = interaction.get("response", "")
            
            # Analyze for calculation errors
            calculation_errors = self._detect_calculation_errors(prompt, response)
            if calculation_errors["count"] > 0:
                error_counts["calculation"] += 1
                severity_sum["calculation"] += calculation_errors["severity"]
            category_counts["calculation"] += 1
            
            # Analyze for planning errors
            planning_errors = self._detect_planning_errors(prompt, response)
            if planning_errors["count"] > 0:
                error_counts["planning"] += 1
                severity_sum["planning"] += planning_errors["severity"]
            category_counts["planning"] += 1
            
            # Analyze for working memory errors
            memory_errors = self._detect_memory_errors(prompt, response)
            if memory_errors["count"] > 0:
                error_counts["working_memory"] += 1
                severity_sum["working_memory"] += memory_errors["severity"]
            category_counts["working_memory"] += 1
        
        # Calculate frequency and average severity
        for category in error_categories:
            if category_counts[category] > 0:
                error_categories[category]["frequency"] = error_counts[category] / category_counts[category]
                if error_counts[category] > 0:
                    error_categories[category]["severity"] = severity_sum[category] / error_counts[category]
        
        return error_categories
    
    def _detect_calculation_errors(self, prompt, response):
        """
        Detect calculation errors in a response
        
        Args:
            prompt: The prompt
            response: The response
            
        Returns:
            Error detection results
        """
        # Extract calculation expressions
        calculations = re.findall(r'(\d+\.?\d*)\s*([+\-*/^])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', response)
        
        error_count = 0
        error_severity = 0.0
        
        for calc in calculations:
            left_operand = float(calc[0])
            operator = calc[1]
            right_operand = float(calc[2])
            result = float(calc[3])
            
            # Calculate the expected result
            expected = None
            if operator == '+':
                expected = left_operand + right_operand
            elif operator == '-':
                expected = left_operand - right_operand
            elif operator == '*':
                expected = left_operand * right_operand
            elif operator == '/':
                expected = left_operand / right_operand if right_operand != 0 else None
            elif operator == '^':
                expected = left_operand ** right_operand
            
            # Check for error
            if expected is not None and abs(expected - result) > 0.001:
                error_count += 1
                # Severity based on relative error
                relative_error = abs(expected - result) / max(abs(expected), 1)
                error_severity += min(1.0, relative_error)
        
        return {
            "count": error_count,
            "severity": error_severity / max(1, error_count)
        }
    
    def _detect_planning_errors(self, prompt, response):
        """
        Detect planning errors in a response
        
        Args:
            prompt: The prompt
            response: The response
            
        Returns:
            Error detection results
        """
        # Look for inconsistencies in multi-step reasoning
        steps = re.split(r'(?i)step\s+\d+:|(?:first|second|third|next|then):', response)
        
        error_count = 0
        error_severity = 0.0
        
        # Check for sequence breaks (steps that don't follow from previous ones)
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            # Look for contradictions
            if "actually" in current_step.lower() or "wait" in current_step.lower() or "mistake" in current_step.lower():
                error_count += 1
                error_severity += 0.5
            
            # Look for repeated work (indicator of planning issues)
            if len(previous_step) > 50 and len(current_step) > 50:
                similarity = self._text_similarity(previous_step, current_step)
                if similarity > 0.7:  # High similarity indicates repeating work
                    error_count += 1
                    error_severity += similarity - 0.7
        
        return {
            "count": error_count,
            "severity": error_severity / max(1, error_count)
        }
    
    def _detect_memory_errors(self, prompt, response):
        """
        Detect working memory errors in a response
        
        Args:
            prompt: The prompt
            response: The response
            
        Returns:
            Error detection results
        """
        # Extract variable assignments and their uses
        assignments = {}
        for var_name, value in re.findall(r'(?:let|set)\s+([a-zA-Z][a-zA-Z0-9]*)\s*=\s*(\d+\.?\d*)', response):
            assignments[var_name] = float(value)
        
        error_count = 0
        error_severity = 0.0
        
        # Check for inconsistent variable use
        for var_name, assigned_value in assignments.items():
            uses = re.findall(fr'{var_name}\s*=\s*(\d+\.?\d*)', response)
            for used_value in uses:
                if abs(float(used_value) - assigned_value) > 0.001:
                    error_count += 1
                    error_severity += min(1.0, abs(float(used_value) - assigned_value) / max(abs(assigned_value), 1))
        
        return {
            "count": error_count,
            "severity": error_severity / max(1, error_count)
        }
    
    def _text_similarity(self, text1, text2):
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word-based Jaccard similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
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
            if not response_text:
                continue
            
            # Hesitation markers
            hesitation_pattern = r'\b(think|probably|might|possibly|perhaps|maybe|not sure|uncertain)\b'
            hesitation = len(re.findall(hesitation_pattern, response_text.lower())) * 0.1
            hesitation = min(1.0, hesitation + 0.2 if "?" in response_text else hesitation)
            
            # Self-correction patterns
            correction_pattern = r'\b(actually|correction|mistake|error|oops|let me recalculate|no wait|sorry)\b'
            corrections = len(re.findall(correction_pattern, response_text.lower())) * 0.2
            
            # Response time factor (longer time suggests lower confidence)
            response_time_factor = min(1.0, result.get("response_time", 1.0) * 0.2)
            
            # Consistency in repeated calculations
            consistency = 0.0
            calculations = re.findall(r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)', response_text)
            unique_results = set()
            for calc in calculations:
                unique_results.add(calc[3])
            if len(calculations) > 1:
                consistency = min(1.0, (len(calculations) - len(unique_results)) / len(calculations))
            
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
        # Determine task difficulty
        task_difficulty = self._estimate_task_difficulty(task)
        
        # Analyze previous response if available
        has_errors = False
        error_types = set()
        if previous_response:
            # Check for calculation errors
            calc_errors = self._detect_calculation_errors("", previous_response)
            if calc_errors["count"] > 0:
                has_errors = True
                error_types.add("calculation")
            
            # Check for planning errors
            plan_errors = self._detect_planning_errors("", previous_response)
            if plan_errors["count"] > 0:
                has_errors = True
                error_types.add("planning")
            
            # Check for memory errors
            mem_errors = self._detect_memory_errors("", previous_response)
            if mem_errors["count"] > 0:
                has_errors = True
                error_types.add("memory")
            
            # Check for explicit confusion
            if any(marker in previous_response.lower() for marker in 
                  ["i'm not sure", "i don't know", "i'm confused", "i'm uncertain"]):
                has_errors = True
                
        # Generate appropriate prompting strategy
        if task_difficulty <= combined_boundary * 0.8:
            # Task well within capabilities - standard CoT
            if has_errors:
                prompt = f"Let's approach this problem step by step, being careful to avoid errors in {', '.join(error_types)}:\n{task}"
            else:
                prompt = f"Please solve this step-by-step:\n{task}"
                
        elif task_difficulty <= combined_boundary * 1.2:
            # Task near boundary - structured decomposition
            if has_errors:
                prompt = f"This problem requires careful thinking. Let's break it down into very small, manageable steps, focusing particularly on accurate {', '.join(error_types)}:\n{task}"
            else:
                prompt = f"Break this problem into smaller steps and solve each step carefully:\n{task}"
                
        else:
            # Task beyond boundary - aggressive scaffolding
            prompt = self._generate_scaffolded_prompt(task, combined_boundary, error_types if has_errors else None)
        
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
        difficulty_score += min(2.0, word_count / 100)
        
        # Count numbers as indicators of calculation complexity
        number_count = len(re.findall(r'\d+', task))
        difficulty_score += min(2.0, number_count * 0.1)
        
        # Check for mathematical operations
        if any(op in task for op in ["+", "-", "*", "/", "^", "="]):
            difficulty_score += 0.5
        
        # Check for logical complexity indicators
        logical_indicators = ["if", "then", "else", "and", "or", "not", "all", "some", "none"]
        logical_count = sum(1 for indicator in logical_indicators if f" {indicator} " in f" {task.lower()} ")
        difficulty_score += min(1.5, logical_count * 0.3)
        
        return difficulty_score
    
    def _generate_scaffolded_prompt(self, task, boundary, error_types=None):
        """
        Generate a heavily scaffolded prompt for difficult tasks
        
        Args:
            task: The reasoning task
            boundary: Boundary estimate
            error_types: Types of errors to address specifically
            
        Returns:
            Scaffolded prompt
        """
        prompt = f"I'll help you break down this complex problem into very small, manageable steps.\n\n"
        prompt += f"Task: {task}\n\n"
        prompt += "Follow these specific steps:\n\n"
        
        # Customize scaffolding based on task characteristics
        if any(keyword in task.lower() for keyword in ["calculate", "compute", "find", "solve"]):
            # Mathematical scaffolding
            prompt += "1. Identify the key variables and what you're solving for.\n"
            prompt += "2. Write down any relevant formulas or equations.\n"
            prompt += "3. Substitute the known values into these equations.\n"
            
            # Add extra care for calculation if it's a known error type
            if error_types and "calculation" in error_types:
                prompt += "4. Perform each calculation separately, double-checking your arithmetic at each step.\n"
                prompt += "5. Verify intermediate results by plugging them back into the equations.\n"
            else:
                prompt += "4. Perform one calculation at a time, showing each step.\n"
                prompt += "5. Double-check your arithmetic at each step.\n"
                
            prompt += "6. State your final answer clearly.\n\n"
        else:
            # General reasoning scaffolding
            prompt += "1. Identify the key information and constraints.\n"
            prompt += "2. Break down the main question into smaller questions.\n"
            
            # Add memory tracking if it's a known error type
            if error_types and "memory" in error_types:
                prompt += "3. Keep track of all values and information in a structured way.\n"
                prompt += "4. Address each sub-question one by one, referencing your tracked information.\n"
            else:
                prompt += "3. Address each sub-question one by one.\n"
            
            # Add planning help if it's a known error type
            if error_types and "planning" in error_types:
                prompt += "4. Before concluding each step, verify it connects logically to the next step.\n"
                prompt += "5. Check that your reasoning chain is complete without gaps.\n"
            else:
                prompt += "4. Connect your findings to form a coherent answer.\n"
                prompt += "5. Verify that your answer addresses the original question.\n"
            
        prompt += "\nNow, let's solve this step by step:"
        
        return prompt
    
    async def process_interaction(self, task, requestor, current_interaction=0):
        """
        Process a single interaction in the DBE framework
        
        Args:
            task: The reasoning task
            requestor: The language model requestor for API calls
            current_interaction: Current interaction number
            
        Returns:
            Adapted prompt and updated boundary estimates
        """
        # Determine if we need to deploy probes
        deploy_probes_now = (current_interaction % self.probe_frequency == 0 or 
                            not self.boundary_estimates)
        
        combined_boundary = 0.0
        
        if deploy_probes_now:
            # Deploy probes to assess boundaries
            print(f"Deploying probes at interaction {current_interaction}...")
            probe_results, _ = await self.deploy_probes(requestor)
            
            # Extract model name from requestor if available
            model_name = getattr(requestor, "model_name", "unknown_model")
            
            # Update boundary estimates
            self.boundary_estimates = self.estimate_boundaries(probe_results, model_name)
            
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
            
            # Save updated estimates
            self._save_boundary_estimates(model_name)
        
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
            "response": None,  # Will be filled with actual response
            "boundary_estimates": self.boundary_estimates.copy(),
            "combined_boundary": combined_boundary
        })
        
        return adapted_prompt, self.boundary_estimates, combined_boundary
    
    def update_interaction_history(self, prompt, response):
        """
        Update interaction history with a response
        
        Args:
            prompt: The prompt
            response: The response
            
        Returns:
            Updated interaction history
        """
        # Update the most recent interaction with the response
        if self.interaction_history and self.interaction_history[-1]["prompt"] == prompt:
            self.interaction_history[-1]["response"] = response
        else:
            # Create a new interaction entry
            self.interaction_history.append({
                "prompt": prompt,
                "response": response,
                "boundary_estimates": self.boundary_estimates.copy() if self.boundary_estimates else {},
                "combined_boundary": self.compute_combined_boundary(self.boundary_estimates)
            })
        
        return self.interaction_history
