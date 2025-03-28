"""
Implementation of Multi-Agent Reasoning Collaboration (MARC)
with Dynamic Boundary Estimation (DBE) integration
"""
import numpy as np
import json
import re
import asyncio
import os
from collections import defaultdict
from utils.request_tool import MMRequestor
from methods.dbe import DBE  # Import DBE for dynamic boundary estimation


class MARC:
    """
    Multi-Agent Reasoning Collaboration (MARC)
    
    MARC leverages multiple specialized agents with complementary reasoning 
    boundaries to collaboratively solve complex tasks.
    
    Now integrated with Dynamic Boundary Estimation (DBE) to continuously
    update agent boundary profiles based on observed performance.
    """
    
    def __init__(self, 
                 agents=None,               # Dictionary of available agents with their boundary profiles
                 max_communication_rounds=5,  # Maximum number of communication rounds
                 api_key=None,              # API key for LLM services
                 api_base=None,             # Base URL for API
                 verbose=False,             # Whether to print detailed process information
                 dbe_params=None            # Parameters for DBE instances
                 ):
        """
        Initialize MARC with agent configuration
        
        Args:
            agents: Dictionary of available agents with their boundary profiles
            max_communication_rounds: Maximum number of communication rounds
            api_key: API key for LLM services
            api_base: Base URL for API
            verbose: Whether to print detailed process information
            dbe_params: Parameters for DBE instances (gamma, probe_frequency, etc.)
        """
        self.agents = agents or {}
        self.max_communication_rounds = max_communication_rounds
        self.api_key = api_key
        self.api_base = api_base
        self.verbose = verbose
        
        # Default DBE parameters if not provided
        self.dbe_params = dbe_params or {
            "gamma": 0.12,               # Error sensitivity parameter
            "probe_frequency": 5,         # How often to deploy probes
            "probe_set_size": 7           # Number of probes per assessment
        }
        
        # Initialize DBE instances for each agent
        self.dbe_instances = {}
        
        # Initialize collaboration state
        self.task_assignments = {}  # Map of subtasks to agents
        self.solution_components = {}  # Partial solutions by subtask
        self.communication_history = []  # Record of agent communications
        self.consensus_votes = defaultdict(dict)  # Votes on critical decisions
        self.subtasks = []  # List of all subtasks
        self.requestors = {}  # Dictionary of requestors for each agent
        self.boundary_updates = {}  # Track boundary updates during collaboration

    def _log(self, message):
        """
        Log messages if verbose mode is enabled
        
        Args:
            message: Message to log
        """
        if self.verbose:
            print(message)
    
    def add_agent(self, agent_id, agent_type, boundary_profile, model_name=None):
        """
        Add an agent to the collaboration framework
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (planner, calculator, verifier, integrator)
            boundary_profile: Dictionary of reasoning boundaries by dimension
            model_name: Name of the underlying language model
            
        Returns:
            Updated agents dictionary
        """
        self.agents[agent_id] = {
            "type": agent_type,
            "boundaries": boundary_profile,
            "model": model_name,
            "current_task": None,
            "status": "idle",
            "boundary_history": {k: [v] for k, v in boundary_profile.items()}  # Track boundary history
        }
        
        # Create a DBE instance for this agent
        results_path = f"experiments/results/dbe_boundaries_{agent_id}.json"
        self.dbe_instances[agent_id] = DBE(
            gamma=self.dbe_params.get("gamma", 0.12),
            probe_frequency=self.dbe_params.get("probe_frequency", 5),
            probe_set_size=self.dbe_params.get("probe_set_size", 7),
            initial_boundary_estimates=boundary_profile.copy(),
            results_path=results_path
        )
        
        self._log(f"Initialized DBE instance for agent {agent_id} with boundary profile: {boundary_profile}")
        
        # Create a requestor for this agent
        model_type = "openai" if model_name and any(name in model_name.lower() for name in ["gpt", "o1", "o3"]) else \
                    "anthropic" if model_name and any(name in model_name.lower() for name in ["claude"]) else \
                    "openai"  # Default to OpenAI
        
        try:
            self.requestors[agent_id] = MMRequestor(
                model_type=model_type,
                model_name=model_name,
                api_key=self.api_key,
                api_base=self.api_base
            )
            self._log(f"Created requestor for agent {agent_id} with model {model_name}")
        except Exception as e:
            self._log(f"Warning: Failed to create requestor for agent {agent_id}: {e}")
        
        self._log(f"Added agent {agent_id} of type {agent_type} with model {model_name}")
        return self.agents
    
    def select_planner_agent(self):
        """
        Select the most suitable agent for planning tasks
        
        Returns:
            Agent ID of selected planner
        """
        best_planner = None
        highest_planning_boundary = -1
        
        for agent_id, agent in self.agents.items():
            planning_boundary = agent["boundaries"].get("planning", 0)
            
            if planning_boundary > highest_planning_boundary:
                highest_planning_boundary = planning_boundary
                best_planner = agent_id
        
        self._log(f"Selected planner agent: {best_planner}")        
        return best_planner
    
    def select_calculator_agent(self):
        """
        Select the most suitable agent for calculation tasks
        
        Returns:
            Agent ID of selected calculator
        """
        best_calculator = None
        highest_calculation_boundary = -1
        
        for agent_id, agent in self.agents.items():
            calculation_boundary = agent["boundaries"].get("calculation", 0)
            
            if calculation_boundary > highest_calculation_boundary:
                highest_calculation_boundary = calculation_boundary
                best_calculator = agent_id
        
        self._log(f"Selected calculator agent: {best_calculator}")
        return best_calculator
    
    def select_verifier_agent(self):
        """
        Select the most suitable agent for verification tasks
        
        Returns:
            Agent ID of selected verifier
        """
        # For verification, prioritize agents with balanced boundaries
        best_verifier = None
        best_score = -1
        
        for agent_id, agent in self.agents.items():
            boundaries = agent["boundaries"]
            # Balance score rewards agents with good boundaries across dimensions
            balance_score = sum(boundaries.values()) * min(boundaries.values()) if boundaries else 0
            
            if balance_score > best_score:
                best_score = balance_score
                best_verifier = agent_id
        
        self._log(f"Selected verifier agent: {best_verifier}")
        return best_verifier
    
    def select_integrator_agent(self):
        """
        Select the most suitable agent for integration tasks
        
        Returns:
            Agent ID of selected integrator
        """
        best_integrator = None
        highest_score = -1
        
        for agent_id, agent in self.agents.items():
            boundaries = agent["boundaries"]
            # Integrators need good working memory and planning
            memory_score = boundaries.get("working_memory", 0)
            planning_score = boundaries.get("planning", 0)
            
            combined_score = memory_score * 0.7 + planning_score * 0.3
            
            if combined_score > highest_score:
                highest_score = combined_score
                best_integrator = agent_id
        
        self._log(f"Selected integrator agent: {best_integrator}")
        return best_integrator
    
    async def deploy_dbe_probes(self, agent_id, requestor=None):
        """
        Deploy DBE probes to assess agent capabilities
        
        Args:
            agent_id: Agent ID to probe
            requestor: Requestor to use for API calls (optional)
            
        Returns:
            Updated boundary estimates
        """
        self._log(f"Deploying DBE probes for agent {agent_id}")
        
        # Get the DBE instance for this agent
        dbe_instance = self.dbe_instances.get(agent_id)
        if not dbe_instance:
            self._log(f"No DBE instance found for agent {agent_id}")
            return None
        
        # Get the requestor for this agent
        if requestor is None:
            requestor = self.requestors.get(agent_id)
            if not requestor:
                self._log(f"No requestor found for agent {agent_id}")
                return None
        
        try:
            # Deploy probes
            probe_results, _ = await dbe_instance.deploy_probes(requestor)
            
            # Extract model name from agent
            model_name = self.agents[agent_id]["model"]
            
            # Update boundary estimates
            updated_boundaries = dbe_instance.estimate_boundaries(probe_results, model_name)
            
            # Analyze error patterns
            error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
            
            # Refine boundary estimates
            refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
            
            # Calibrate confidence
            confidence_factors = dbe_instance.calibrate_confidence(probe_results)
            
            # Apply confidence calibration
            calibrated_estimates = dbe_instance.calibrate_boundary_estimates(refined_estimates, confidence_factors)
            
            # Update agent boundaries with calibrated estimates
            self.update_agent_boundaries(agent_id, calibrated_estimates)
            
            self._log(f"Updated boundaries for agent {agent_id}: {calibrated_estimates}")
            return calibrated_estimates
            
        except Exception as e:
            self._log(f"Error deploying DBE probes for agent {agent_id}: {e}")
            return None
    
    def update_agent_boundaries(self, agent_id, boundary_estimates):
        """
        Update an agent's boundary profile with DBE estimates
        
        Args:
            agent_id: Agent ID to update
            boundary_estimates: New boundary estimates
            
        Returns:
            Updated agent boundary profile
        """
        if agent_id not in self.agents:
            self._log(f"Agent {agent_id} not found")
            return None
        
        # Update agent boundaries
        agent = self.agents[agent_id]
        old_boundaries = agent["boundaries"].copy()
        
        # Only update dimensions that are in both estimates and original boundaries
        for dim, value in boundary_estimates.items():
            if dim in agent["boundaries"]:
                agent["boundaries"][dim] = value
                
                # Track boundary history
                if dim in agent["boundary_history"]:
                    agent["boundary_history"][dim].append(value)
                else:
                    agent["boundary_history"][dim] = [value]
        
        # Record this update
        self.boundary_updates[agent_id] = {
            "timestamp": len(self.boundary_updates.get(agent_id, [])),
            "old_boundaries": old_boundaries,
            "new_boundaries": agent["boundaries"].copy()
        }
        
        self._log(f"Updated boundaries for agent {agent_id}: {agent['boundaries']}")
        return agent["boundaries"]
    
    async def decompose_task(self, task, planner_agent_id):
        """
        Decompose the main task into subtasks using the planner agent
        
        Args:
            task: The main reasoning task
            planner_agent_id: Agent ID of the selected planner
            
        Returns:
            List of subtasks
        """
        self._log(f"Decomposing task using planner agent {planner_agent_id}")
        
        # Get the planner agent's model
        planner = self.agents[planner_agent_id]
        planner_model = planner["model"]
        
        # Prepare the prompt for task decomposition
        decomposition_prompt = f"""
        You are a task planning expert. Your role is to break down a complex reasoning task into smaller, manageable subtasks.
        
        For each subtask, provide:
        1. A unique ID (e.g., "parse", "formulate", "solve")
        2. A clear description of what needs to be done
        3. Dependencies (which subtasks must be completed before this one)
        
        Format your response as a JSON list with each subtask having these fields:
        - "id": unique identifier
        - "description": clear description of the subtask
        - "dependencies": list of subtask IDs that must be completed first (empty list if none)
        
        Task to decompose: {task}
        
        Response format example:
        [
            {{"id": "parse", "description": "Parse the problem statement", "dependencies": []}},
            {{"id": "formulate", "description": "Formulate the equations needed", "dependencies": ["parse"]}},
            ...
        ]
        
        Now please decompose the task:
        """
        
        # Make the API call to the planner agent
        try:
            requestor = self.requestors.get(planner_agent_id)
            if not requestor:
                self._log(f"No requestor found for agent {planner_agent_id}, creating a new one")
                model_type = "openai" if planner_model and any(name in planner_model.lower() for name in ["gpt", "o1", "o3"]) else \
                            "anthropic" if planner_model and any(name in planner_model.lower() for name in ["claude"]) else \
                            "openai"
                requestor = MMRequestor(
                    model_type=model_type,
                    model_name=planner_model,
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                self.requestors[planner_agent_id] = requestor
            
            response = await requestor.request(decomposition_prompt, temperature=0.3, max_tokens=2048)
            
            # Extract the response text
            response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else response
            
            # Extract JSON from the response
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                subtasks = json.loads(json_str)
            else:
                # If JSON can't be extracted, try to parse the whole response
                try:
                    subtasks = json.loads(response_text)
                except:
                    # If all parsing fails, create a default decomposition
                    self._log("Failed to parse subtasks JSON, using default decomposition")
                    task_type = self._identify_task_type(task)
                    if task_type == "mathematical":
                        subtasks = self._default_mathematical_decomposition(task)
                    elif task_type == "logical":
                        subtasks = self._default_logical_decomposition(task)
                    else:
                        subtasks = self._default_general_decomposition(task)
            
            # Validate and fix subtasks format
            subtasks = self._validate_subtasks(subtasks, task)
            
            # Save subtasks for future reference
            self.subtasks = subtasks
            self._log(f"Task decomposed into {len(subtasks)} subtasks")
            
            # Update DBE interaction history for the planner
            if planner_agent_id in self.dbe_instances:
                dbe_instance = self.dbe_instances[planner_agent_id]
                dbe_instance.update_interaction_history(decomposition_prompt, response_text)
                
                # Update boundary estimates after this interaction
                error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
                refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
                self.update_agent_boundaries(planner_agent_id, refined_estimates)
            
            return subtasks
            
        except Exception as e:
            self._log(f"Error in task decomposition: {str(e)}")
            # Fallback to default decomposition based on task type
            task_type = self._identify_task_type(task)
            if task_type == "mathematical":
                subtasks = self._default_mathematical_decomposition(task)
            elif task_type == "logical":
                subtasks = self._default_logical_decomposition(task)
            else:
                subtasks = self._default_general_decomposition(task)
            
            # Save subtasks for future reference
            self.subtasks = subtasks
            self._log(f"Using default decomposition with {len(subtasks)} subtasks")
            
            return subtasks
    
    def _validate_subtasks(self, subtasks, task):
        """
        Validate and fix subtasks format
        
        Args:
            subtasks: List of subtasks to validate
            task: Original task for context
            
        Returns:
            Validated subtasks
        """
        valid_subtasks = []
        
        # Check if subtasks is a list
        if not isinstance(subtasks, list):
            self._log("Subtasks is not a list, using default decomposition")
            task_type = self._identify_task_type(task)
            return self._get_default_decomposition(task_type, task)
        
        # Check each subtask for required fields
        for i, subtask in enumerate(subtasks):
            if not isinstance(subtask, dict):
                self._log(f"Subtask {i} is not a dictionary, skipping")
                continue
                
            # Ensure required fields exist
            if "id" not in subtask:
                subtask["id"] = f"subtask_{i}"
            
            if "description" not in subtask:
                subtask["description"] = f"Subtask {i} for task: {task[:50]}..."
                
            if "dependencies" not in subtask:
                subtask["dependencies"] = []
                
            # Ensure dependencies is a list
            if not isinstance(subtask["dependencies"], list):
                subtask["dependencies"] = []
                
            valid_subtasks.append(subtask)
        
        # If no valid subtasks, use default decomposition
        if not valid_subtasks:
            self._log("No valid subtasks found, using default decomposition")
            task_type = self._identify_task_type(task)
            return self._get_default_decomposition(task_type, task)
            
        return valid_subtasks
    
    def _get_default_decomposition(self, task_type, task):
        """
        Get default decomposition based on task type
        
        Args:
            task_type: Type of task
            task: Original task
            
        Returns:
            Default decomposition
        """
        if task_type == "mathematical":
            return self._default_mathematical_decomposition(task)
        elif task_type == "logical":
            return self._default_logical_decomposition(task)
        else:
            return self._default_general_decomposition(task)
    
    def _identify_task_type(self, task):
        """
        Identify the type of reasoning task
        
        Args:
            task: The reasoning task
            
        Returns:
            Task type identifier
        """
        # In a real implementation, this would use NLP techniques
        # For this implementation, we'll use a simple keyword-based approach
        
        if any(keyword in task.lower() for keyword in ["calculate", "compute", "math", "equation", "solve", "arithmetic"]):
            return "mathematical"
        elif any(keyword in task.lower() for keyword in ["logic", "deduce", "infer", "prove", "conclude", "syllogism"]):
            return "logical"
        else:
            return "general"
    
    def _default_mathematical_decomposition(self, task):
        """
        Decompose a mathematical task into default subtasks
        
        Args:
            task: The mathematical reasoning task
            
        Returns:
            List of subtasks
        """
        return [
            {"id": "parse", "description": f"Parse the mathematical problem: {task}", "dependencies": []},
            {"id": "formulate", "description": "Formulate the mathematical equations needed", "dependencies": ["parse"]},
            {"id": "solve", "description": "Solve the equations step by step", "dependencies": ["formulate"]},
            {"id": "verify", "description": "Verify the solution by checking steps and substituting back", "dependencies": ["solve"]},
            {"id": "explain", "description": "Explain the final answer in context of the original problem", "dependencies": ["verify"]}
        ]
    
    def _default_logical_decomposition(self, task):
        """
        Decompose a logical reasoning task into default subtasks
        
        Args:
            task: The logical reasoning task
            
        Returns:
            List of subtasks
        """
        return [
            {"id": "premises", "description": f"Identify the key premises in the problem: {task}", "dependencies": []},
            {"id": "rules", "description": "Identify logical rules or principles that apply", "dependencies": ["premises"]},
            {"id": "infer", "description": "Draw logical inferences step by step", "dependencies": ["premises", "rules"]},
            {"id": "validate", "description": "Check for logical fallacies or contradictions", "dependencies": ["infer"]},
            {"id": "conclude", "description": "Form a final conclusion based on the validated reasoning", "dependencies": ["validate"]}
        ]
    
    def _default_general_decomposition(self, task):
        """
        Decompose a general reasoning task into default subtasks
        
        Args:
            task: The general reasoning task
            
        Returns:
            List of subtasks
        """
        return [
            {"id": "analyze", "description": f"Analyze the key components of the problem: {task}", "dependencies": []},
            {"id": "research", "description": "Identify relevant facts and context", "dependencies": ["analyze"]},
            {"id": "structure", "description": "Structure an approach to address the problem", "dependencies": ["analyze", "research"]},
            {"id": "reason", "description": "Apply reasoning to draw conclusions", "dependencies": ["structure"]},
            {"id": "summarize", "description": "Summarize findings and provide a final answer", "dependencies": ["reason"]}
        ]
    
    def estimate_task_difficulty(self, subtask):
        """
        Estimate difficulty vector for a subtask
        
        Args:
            subtask: The subtask to assess
            
        Returns:
            Difficulty vector across reasoning dimensions
        """
        subtask_id = subtask["id"]
        description = subtask["description"]
        
        # Base difficulty values by subtask type
        difficulty_templates = {
            "parse": {"calculation": 1.0, "planning": 2.0, "working_memory": 1.5},
            "formulate": {"calculation": 2.0, "planning": 3.0, "working_memory": 2.5},
            "solve": {"calculation": 4.0, "planning": 2.0, "working_memory": 3.0},
            "verify": {"calculation": 3.0, "planning": 1.5, "working_memory": 4.0},
            "explain": {"calculation": 1.0, "planning": 2.5, "working_memory": 3.0},
            
            "premises": {"calculation": 1.0, "planning": 2.5, "working_memory": 2.0},
            "rules": {"calculation": 1.5, "planning": 3.0, "working_memory": 2.5},
            "infer": {"calculation": 2.0, "planning": 4.0, "working_memory": 3.5},
            "validate": {"calculation": 2.5, "planning": 3.0, "working_memory": 4.0},
            "conclude": {"calculation": 1.5, "planning": 2.0, "working_memory": 3.0},
            
            "analyze": {"calculation": 1.5, "planning": 3.0, "working_memory": 2.0},
            "research": {"calculation": 1.0, "planning": 2.0, "working_memory": 3.0},
            "structure": {"calculation": 1.5, "planning": 4.0, "working_memory": 2.5},
            "reason": {"calculation": 2.5, "planning": 3.5, "working_memory": 3.0},
            "summarize": {"calculation": 1.0, "planning": 2.0, "working_memory": 3.5}
        }
        
        # Start with template difficulty based on subtask ID
        difficulty = difficulty_templates.get(subtask_id, {"calculation": 2.0, "planning": 2.0, "working_memory": 2.0})
        
        # Adjust based on task description length and complexity
        description_length = len(description.split())
        complexity_factor = max(0.8, min(1.5, description_length / 20))
        
        # Apply complexity factor
        for dim in difficulty:
            difficulty[dim] *= complexity_factor
            
            # Additional adjustments based on keywords
            if "step by step" in description.lower():
                difficulty["planning"] *= 1.2
            if "calculate" in description.lower() or "compute" in description.lower():
                difficulty["calculation"] *= 1.3
            if "remember" in description.lower() or "track" in description.lower():
                difficulty["working_memory"] *= 1.2
                
        return difficulty
    
    def measure_boundary_alignment(self, agent_boundaries, task_difficulty):
        """
        Measure how well an agent's boundaries align with task requirements
        
        Args:
            agent_boundaries: Agent's boundary profile
            task_difficulty: Task difficulty vector
            
        Returns:
            Alignment score (higher is better)
        """
        if not agent_boundaries or not task_difficulty:
            return 0.0
        
        alignment_score = 0.0
        dimensions_count = 0
        
        for dim in task_difficulty:
            if dim in agent_boundaries:
                # Higher score when boundary is comfortably above difficulty
                margin = agent_boundaries[dim] - task_difficulty[dim]
                if margin > 0:
                    # Positive margin (boundary > difficulty) is good
                    alignment_score += 1.0 + min(1.0, margin / task_difficulty[dim])
                else:
                    # Negative margin (boundary < difficulty) is bad
                    alignment_score += max(0.0, 1.0 + margin / task_difficulty[dim])
                
                dimensions_count += 1
        
        # Average across dimensions
        return alignment_score / dimensions_count if dimensions_count > 0 else 0.0
    
    def assign_tasks(self, subtasks):
        """
        Assign subtasks to agents based on boundary alignment
        
        Args:
            subtasks: List of subtasks to assign
            
        Returns:
            Dictionary mapping subtask IDs to agent IDs
        """
        task_assignments = {}
        
        for subtask in subtasks:
            # Estimate task difficulty
            difficulty = self.estimate_task_difficulty(subtask)
            
            # Find best agent
            best_agent = None
            best_alignment = -1
            
            for agent_id, agent in self.agents.items():
                # Use the most up-to-date boundary estimates
                alignment = self.measure_boundary_alignment(agent["boundaries"], difficulty)
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    best_agent = agent_id
            
            if best_agent:
                task_assignments[subtask["id"]] = {
                    "agent_id": best_agent,
                    "alignment": best_alignment,
                    "subtask": subtask
                }
                
                # Update agent status
                self.agents[best_agent]["status"] = "assigned"
                self.agents[best_agent]["current_task"] = subtask["id"]
                
                self._log(f"Assigned subtask {subtask['id']} to agent {best_agent} (alignment: {best_alignment:.2f})")
        
        return task_assignments
    
    async def process_subtask(self, agent_id, subtask):
        """
        Process a subtask with a specific agent
        
        Args:
            agent_id: Agent ID assigned to the task
            subtask: The subtask to process
            
        Returns:
            Subtask result and status
        """
        self._log(f"Processing subtask {subtask['id']} with agent {agent_id}")
        
        difficulty = self.estimate_task_difficulty(subtask)
        agent = self.agents[agent_id]
        
        # Check if task is beyond agent's boundaries
        beyond_boundary = False
        for dim, diff in difficulty.items():
            if dim in agent["boundaries"] and diff > agent["boundaries"][dim] * 1.2:
                beyond_boundary = True
                self._log(f"Subtask {subtask['id']} exceeds agent {agent_id}'s {dim} boundary")
                break
        
        if beyond_boundary:
            # Agent struggles with this task
            return {
                "status": "boundary_reached",
                "subtask_id": subtask["id"],
                "agent_id": agent_id,
                "result": None,
                "confidence": 0.3,
                "message": f"This task exceeds my capabilities in one or more reasoning dimensions."
            }
        
        # Get relevant solution components for dependencies
        context = self._get_dependency_context(subtask)
        
        # Prepare the prompt
        prompt = self._create_subtask_prompt(subtask, agent["type"], context)
        
        # Make the API call
        try:
            requestor = self.requestors.get(agent_id)
            if not requestor:
                self._log(f"No requestor found for agent {agent_id}, creating a new one")
                model_type = "openai" if agent["model"] and any(name in agent["model"].lower() for name in ["gpt", "o1", "o3"]) else \
                            "anthropic" if agent["model"] and any(name in agent["model"].lower() for name in ["claude"]) else \
                            "openai"
                requestor = MMRequestor(
                    model_type=model_type,
                    model_name=agent["model"],
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                self.requestors[agent_id] = requestor
            
            response = await requestor.request(prompt, temperature=0.3, max_tokens=2048)
            
            # Extract the response text
            response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else response
            
            # Update DBE interaction history and boundary estimates
            if agent_id in self.dbe_instances:
                dbe_instance = self.dbe_instances[agent_id]
                dbe_instance.update_interaction_history(prompt, response_text)
                
                # Analyze the interaction and update boundary estimates
                error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
                refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
                
                # Update agent boundaries with refined estimates
                self.update_agent_boundaries(agent_id, refined_estimates)
            
            # Analyze confidence in the response
            confidence = self._analyze_confidence(response_text)
            
            # If confidence is too low, consider it as boundary reached
            if confidence < 0.4:
                return {
                    "status": "boundary_reached",
                    "subtask_id": subtask["id"],
                    "agent_id": agent_id,
                    "result": response_text,
                    "confidence": confidence,
                    "message": f"Low confidence in my response indicates I may have reached my reasoning boundary."
                }
            
            # Return successful result
            return {
                "status": "completed",
                "subtask_id": subtask["id"],
                "agent_id": agent_id,
                "result": response_text,
                "confidence": confidence,
                "message": "Task completed successfully."
            }
            
        except Exception as e:
            self._log(f"Error processing subtask {subtask['id']}: {str(e)}")
            return {
                "status": "error",
                "subtask_id": subtask["id"],
                "agent_id": agent_id,
                "result": None,
                "confidence": 0.0,
                "message": f"Error processing task: {str(e)}"
            }
    
    def _create_subtask_prompt(self, subtask, agent_type, context):
        """
        Create a prompt for a subtask based on agent type
        
        Args:
            subtask: The subtask to process
            agent_type: Type of agent
            context: Context from dependencies
            
        Returns:
            Prompt for the subtask
        """
        # Add agent-specific role and task instructions
        if agent_type == "planner":
            role = "You are a strategic planning expert who excels at breaking down complex problems into clear steps."
        elif agent_type == "calculator":
            role = "You are a mathematical calculation expert who performs precise computations with high accuracy."
        elif agent_type == "verifier":
            role = "You are a verification expert who carefully checks work for errors and ensures correctness."
        elif agent_type == "integrator":
            role = "You are an integration expert who combines diverse pieces of information into coherent solutions."
        else:
            role = "You are a reasoning expert who solves complex problems methodically."
        
        # Create the prompt
        prompt = f"{role}\n\n"
        prompt += f"Task: {subtask['description']}\n\n"
        
        # Add dependencies context
        if context:
            prompt += "Previous Work:\n"
            for dep_id, dep_result in context.items():
                prompt += f"From {dep_id}: {dep_result}\n\n"
        
        # Add specific instructions based on agent type
        if agent_type == "calculator":
            prompt += "Show all calculation steps clearly. Double-check your work before providing the final answer.\n"
        elif agent_type == "verifier":
            prompt += "Carefully verify the work done so far. Check for errors, inconsistencies, or omissions.\n"
        elif agent_type == "integrator":
            prompt += "Synthesize the information from previous steps into a coherent solution.\n"
        
        prompt += "\nProvide your response:"
        
        return prompt
    
    def _get_dependency_context(self, subtask):
        """
        Get context from dependencies for a subtask
        
        Args:
            subtask: The subtask to get context for
            
        Returns:
            Dictionary of dependency results
        """
        context = {}
        
        for dep_id in subtask.get("dependencies", []):
            if dep_id in self.solution_components:
                result = self.solution_components[dep_id].get("result", "")
                if result:
                    context[dep_id] = result
        
        return context
    
    def _analyze_confidence(self, response_text):
        """
        Analyze confidence level in a response
        
        Args:
            response_text: Text response to analyze
            
        Returns:
            Confidence score (0-1)
        """
        # Confidence reducers
        hesitation_markers = [
            "i think", "probably", "might", "could be", "possibly", "seems like",
            "not sure", "guess", "assume", "perhaps", "approximate", "roughly"
        ]
        
        # Confidence boosters
        certainty_markers = [
            "definitely", "certainly", "clearly", "exactly", "precisely", "absolutely",
            "undoubtedly", "confident", "sure", "without doubt", "evidently"
        ]
        
        # Count markers
        hesitation_count = sum(response_text.lower().count(marker) for marker in hesitation_markers)
        certainty_count = sum(response_text.lower().count(marker) for marker in certainty_markers)
        
        # Check for question marks (indicates uncertainty)
        question_marks = response_text.count("?")
        
        # Base confidence score
        base_confidence = 0.7
        
        # Adjust based on markers
        confidence = base_confidence - (hesitation_count * 0.05) + (certainty_count * 0.05) - (question_marks * 0.1)
        
        # Keep in range [0.1, 0.95]
        return max(0.1, min(0.95, confidence))
    
    def format_communication(self, sender_id, receiver_id, subtask_id, content, content_type="result"):
        """
        Format a communication message between agents
        
        Args:
            sender_id: Agent ID of the sender
            receiver_id: Agent ID of the receiver
            subtask_id: ID of the subtask the message relates to
            content: Message content
            content_type: Type of message content
            
        Returns:
            Formatted message
        """
        return {
            "sender": sender_id,
            "receiver": receiver_id,
            "timestamp": len(self.communication_history),
            "subtask_id": subtask_id,
            "content_type": content_type,
            "content": content
        }
    
    def broadcast_update(self, sender_id, subtask_id, result):
        """
        Broadcast an update to all agents
        
        Args:
            sender_id: Agent ID sending the update
            subtask_id: Subtask ID the update relates to
            result: Result to broadcast
            
        Returns:
            List of sent messages
        """
        messages = []
        
        for agent_id in self.agents:
            if agent_id != sender_id:
                message = self.format_communication(
                    sender_id, 
                    agent_id, 
                    subtask_id, 
                    result, 
                    content_type="update"
                )
                
                messages.append(message)
                self.communication_history.append(message)
        
        return messages
    
    def request_assistance(self, agent_id, subtask_id, difficulty):
        """
        Request assistance when a task exceeds agent's boundaries
        
        Args:
            agent_id: Agent ID requesting assistance
            subtask_id: Subtask ID needing assistance
            difficulty: Difficulty vector of the subtask
            
        Returns:
            Collaboration plan
        """
        self._log(f"Agent {agent_id} requesting assistance for subtask {subtask_id}")
        
        # Find agents with complementary strengths
        assistance_plan = []
        
        # Identify the most challenging dimension
        challenging_dim = max(difficulty, key=difficulty.get)
        
        # Find agent strong in this dimension
        best_assistant = None
        highest_boundary = -1
        
        for assistant_id, assistant in self.agents.items():
            if assistant_id != agent_id:
                boundary = assistant["boundaries"].get(challenging_dim, 0)
                
                if boundary > highest_boundary:
                    highest_boundary = boundary
                    best_assistant = assistant_id
        
        if best_assistant:
            assistance_plan.append({
                "subtask_id": subtask_id,
                "lead_agent": agent_id,
                "assistant_agent": best_assistant,
                "focus_dimension": challenging_dim,
                "collaboration_type": "assisted"
            })
            
            # Create collaboration message
            message = self.format_communication(
                agent_id,
                best_assistant,
                subtask_id,
                f"Request assistance with {challenging_dim} aspects of subtask {subtask_id}.",
                content_type="assistance_request"
            )
            
            self.communication_history.append(message)
            self._log(f"Agent {agent_id} requested assistance from agent {best_assistant} for {challenging_dim}")
        
        return assistance_plan
    
    async def process_collaborative_subtask(self, lead_agent_id, assistant_agent_id, subtask_id, focus_dimension):
        """
        Process a subtask collaboratively with two agents
        
        Args:
            lead_agent_id: ID of the lead agent
            assistant_agent_id: ID of the assistant agent
            subtask_id: ID of the subtask
            focus_dimension: Dimension to focus collaboration on
            
        Returns:
            Collaborative result
        """
        self._log(f"Processing subtask {subtask_id} collaboratively with agents {lead_agent_id} and {assistant_agent_id}")
        
        # Get the subtask
        subtask = next((s for s in self.subtasks if s["id"] == subtask_id), None)
        if not subtask:
            self._log(f"Subtask {subtask_id} not found")
            return {
                "status": "error",
                "subtask_id": subtask_id,
                "agent_id": f"{lead_agent_id}+{assistant_agent_id}",
                "result": "Subtask not found",
                "confidence": 0.0,
                "message": f"Subtask {subtask_id} not found"
            }
        
        # Get agents
        lead_agent = self.agents.get(lead_agent_id)
        assistant_agent = self.agents.get(assistant_agent_id)
        
        if not lead_agent or not assistant_agent:
            self._log(f"One or both agents not found")
            return {
                "status": "error",
                "subtask_id": subtask_id,
                "agent_id": f"{lead_agent_id}+{assistant_agent_id}",
                "result": "Agent not found",
                "confidence": 0.0,
                "message": f"One or both agents not found"
            }
            
        # Get dependency context
        context = self._get_dependency_context(subtask)
        
        # First, have the assistant agent focus on the challenging dimension
        assistant_prompt = f"""
        You are a {focus_dimension} expert. A collaborator needs help with this aspect of a problem.
        
        Task: {subtask['description']}
        
        Focus on providing help specifically with the {focus_dimension} aspects of this problem.
        
        Previous Work:
        """
        
        if context:
            for dep_id, dep_result in context.items():
                assistant_prompt += f"From {dep_id}: {dep_result}\n\n"
        
        assistant_prompt += f"\nProvide your response focusing on the {focus_dimension} aspects:"
        
        try:
            # Get the assistant's response
            assistant_requestor = self.requestors.get(assistant_agent_id)
            if not assistant_requestor:
                self._log(f"No requestor found for assistant agent {assistant_agent_id}, creating a new one")
                model_type = "openai" if assistant_agent["model"] and any(name in assistant_agent["model"].lower() for name in ["gpt", "o1", "o3"]) else \
                            "anthropic" if assistant_agent["model"] and any(name in assistant_agent["model"].lower() for name in ["claude"]) else \
                            "openai"
                assistant_requestor = MMRequestor(
                    model_type=model_type,
                    model_name=assistant_agent["model"],
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                self.requestors[assistant_agent_id] = assistant_requestor
            
            assistant_response = await assistant_requestor.request(assistant_prompt, temperature=0.3, max_tokens=2048)
            assistant_text = assistant_response[-1]["content"][0]["text"] if isinstance(assistant_response, list) else assistant_response
            
            # Update DBE interaction history for the assistant
            if assistant_agent_id in self.dbe_instances:
                dbe_instance = self.dbe_instances[assistant_agent_id]
                dbe_instance.update_interaction_history(assistant_prompt, assistant_text)
                
                # Update boundary estimates for the assistant
                error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
                refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
                self.update_agent_boundaries(assistant_agent_id, refined_estimates)
            
            # Now, have the lead agent complete the task with the assistant's input
            lead_prompt = f"""
            You are working on solving this task: {subtask['description']}
            
            Another expert has provided assistance with the {focus_dimension} aspects:
            
            EXPERT ASSISTANCE:
            {assistant_text}
            
            Previous Work:
            """
            
            if context:
                for dep_id, dep_result in context.items():
                    lead_prompt += f"From {dep_id}: {dep_result}\n\n"
            
            lead_prompt += "\nUsing this assistance, complete the task:"
            
            # Get the lead agent's final response
            lead_requestor = self.requestors.get(lead_agent_id)
            if not lead_requestor:
                self._log(f"No requestor found for lead agent {lead_agent_id}, creating a new one")
                model_type = "openai" if lead_agent["model"] and any(name in lead_agent["model"].lower() for name in ["gpt", "o1", "o3"]) else \
                            "anthropic" if lead_agent["model"] and any(name in lead_agent["model"].lower() for name in ["claude"]) else \
                            "openai"
                lead_requestor = MMRequestor(
                    model_type=model_type,
                    model_name=lead_agent["model"],
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                self.requestors[lead_agent_id] = lead_requestor
            
            lead_response = await lead_requestor.request(lead_prompt, temperature=0.3, max_tokens=2048)
            lead_text = lead_response[-1]["content"][0]["text"] if isinstance(lead_response, list) else lead_response
            
            # Update DBE interaction history for the lead agent
            if lead_agent_id in self.dbe_instances:
                dbe_instance = self.dbe_instances[lead_agent_id]
                dbe_instance.update_interaction_history(lead_prompt, lead_text)
                
                # Update boundary estimates for the lead agent
                error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
                refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
                self.update_agent_boundaries(lead_agent_id, refined_estimates)
            
            # Combine the results
            combined_result = f"Collaborative solution to subtask {subtask_id}:\n\n{lead_text}"
            
            # Analyze confidence
            confidence = self._analyze_confidence(lead_text)
            
            return {
                "status": "completed",
                "subtask_id": subtask_id,
                "agent_id": f"{lead_agent_id}+{assistant_agent_id}",
                "result": combined_result,
                "confidence": confidence,
                "message": "Task completed through collaboration."
            }
            
        except Exception as e:
            self._log(f"Error in collaborative processing: {str(e)}")
            return {
                "status": "error",
                "subtask_id": subtask_id,
                "agent_id": f"{lead_agent_id}+{assistant_agent_id}",
                "result": f"Error in collaborative processing: {str(e)}",
                "confidence": 0.3,
                "message": "Collaboration encountered an error."
            }
    
    async def reach_consensus(self, subtask_id, question, options):
        """
        Reach consensus through weighted voting
        
        Args:
            subtask_id: Subtask ID the consensus relates to
            question: The decision question
            options: Possible options
            
        Returns:
            Consensus decision
        """
        self._log(f"Reaching consensus on subtask {subtask_id}")
        votes = self.consensus_votes.get(subtask_id, {})
        
        if not votes:
            # Get votes from agents
            for agent_id, agent in self.agents.items():
                # Calculate weight based on boundary alignment with task
                subtask = next((s for s in self.subtasks if s["id"] == subtask_id), None)
                if subtask:
                    difficulty = self.estimate_task_difficulty(subtask)
                    alignment = self.measure_boundary_alignment(agent["boundaries"], difficulty)
                    weight = max(0.1, alignment)  # Minimum weight to ensure all agents have some voice
                else:
                    weight = 1.0  # Default weight
                
                # Confidence based on agent's capability profile
                confidence = self._calculate_agent_confidence(agent_id, subtask_id)
                
                # Create a voting prompt
                voting_prompt = f"""
                You need to vote on how to handle a challenging reasoning task.
                
                Question: {question}
                
                Options:
                """
                
                for i, option in enumerate(options):
                    voting_prompt += f"{i+1}. {option}\n"
                
                voting_prompt += "\nSelect the option number that you think is best:"
                
                try:
                    # Get the agent's vote
                    requestor = self.requestors.get(agent_id)
                    if not requestor:
                        self._log(f"No requestor found for agent {agent_id}, creating a new one")
                        model_type = "openai" if agent["model"] and any(name in agent["model"].lower() for name in ["gpt", "o1", "o3"]) else \
                                    "anthropic" if agent["model"] and any(name in agent["model"].lower() for name in ["claude"]) else \
                                    "openai"
                        requestor = MMRequestor(
                            model_type=model_type,
                            model_name=agent["model"],
                            api_key=self.api_key,
                            api_base=self.api_base
                        )
                        self.requestors[agent_id] = requestor
                    
                    response = await requestor.request(voting_prompt, temperature=0.3, max_tokens=256)
                    response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else response
                    
                    # Update DBE interaction history for this agent
                    if agent_id in self.dbe_instances:
                        dbe_instance = self.dbe_instances[agent_id]
                        dbe_instance.update_interaction_history(voting_prompt, response_text)
                    
                    # Extract the option number from the response
                    option_match = re.search(r'\b([1-9])\b', response_text)
                    option_num = int(option_match.group(1)) if option_match else 1
                    
                    # Adjust if out of range
                    option_num = max(1, min(option_num, len(options)))
                    
                    # Get the selected option
                    preference = options[option_num - 1]
                    
                    votes[agent_id] = {
                        "option": preference,
                        "weight": weight,
                        "confidence": confidence
                    }
                    
                    self._log(f"Agent {agent_id} voted for: {preference}")
                    
                except Exception as e:
                    self._log(f"Error getting vote from agent {agent_id}: {str(e)}")
                    # Default vote in case of error
                    votes[agent_id] = {
                        "option": options[0],
                        "weight": weight,
                        "confidence": confidence
                    }
                
            # Record in consensus votes
            self.consensus_votes[subtask_id] = votes
        
        # Tally weighted votes according to equation (15) in the paper
        option_scores = defaultdict(float)
        
        for agent_id, vote in votes.items():
            option = vote["option"]
            weight = vote["weight"]
            confidence = vote["confidence"]
            
            # Weight by both agent weight and confidence as per equation (15)
            option_scores[option] += weight * confidence
        
        # Find option with highest score
        if option_scores:
            consensus = max(option_scores.items(), key=lambda x: x[1])
            total_score = sum(option_scores.values())
            
            return {
                "option": consensus[0],
                "score": consensus[1],
                "agreement_level": consensus[1] / total_score if total_score > 0 else 0,
                "vote_distribution": {k: v / total_score for k, v in option_scores.items()} if total_score > 0 else {}
            }
        
        return None
    
    def _calculate_agent_confidence(self, agent_id, subtask_id):
        """
        Calculate agent's confidence for a specific subtask
        
        Args:
            agent_id: Agent ID
            subtask_id: Subtask ID
            
        Returns:
            Confidence score (0-1)
        """
        # Get agent boundaries
        agent = self.agents.get(agent_id, {})
        boundaries = agent.get("boundaries", {})
        
        # Get subtask
        subtask = next((s for s in self.subtasks if s["id"] == subtask_id), None)
        if not subtask:
            return 0.5  # Default confidence
        
        # Get subtask difficulty
        difficulty = self.estimate_task_difficulty(subtask)
        
        # Calculate confidence based on boundary vs difficulty
        confidence = 0.5  # Base confidence
        for dim, diff in difficulty.items():
            if dim in boundaries:
                # Higher boundary relative to difficulty means higher confidence
                boundary_ratio = boundaries[dim] / (diff + 0.001)  # Avoid division by zero
                if boundary_ratio > 1.5:
                    confidence += 0.2
                elif boundary_ratio > 1.0:
                    confidence += 0.1
                elif boundary_ratio < 0.7:
                    confidence -= 0.2
        
        # Clamp confidence to 0.1-0.9 range
        return max(0.1, min(0.9, confidence))
    
    async def synthesize_solution(self, solution_components):
        """
        Synthesize final solution from component parts
        
        Args:
            solution_components: Dictionary of partial solutions by subtask
            
        Returns:
            Synthesized complete solution
        """
        self._log("Synthesizing final solution")
        
        if not solution_components:
            return "No solution components available."
        
        # Select integrator agent
        integrator_id = self.select_integrator_agent()
        
        if not integrator_id:
            # Fallback to any available agent
            integrator_id = next(iter(self.agents.keys())) if self.agents else None
            
        if not integrator_id:
            self._log("No integrator agent available")
            return "No integrator agent available for synthesis."
        
        # Prepare the components for the prompt
        components_text = ""
        for subtask_id, component in solution_components.items():
            if isinstance(component, dict) and "result" in component:
                components_text += f"Result from subtask {subtask_id}:\n{component['result']}\n\n"
            elif isinstance(component, str):
                components_text += f"Result from subtask {subtask_id}:\n{component}\n\n"
        
        # Create the integration prompt
        integration_prompt = f"""
        You are a solution integrator expert. Your task is to synthesize multiple partial solutions into a coherent final answer.
        
        Component solutions:
        {components_text}
        
        Please synthesize these components into a complete, coherent solution.
        Your synthesis should include:
        1. A clear integration of all relevant information from the components
        2. A logical flow between the different parts
        3. A final answer that addresses the original problem
        
        Provide your synthesized solution:
        """
        
        try:
            # Get the integrator agent's response
            requestor = self.requestors.get(integrator_id)
            if not requestor:
                self._log(f"No requestor found for integrator agent {integrator_id}, creating a new one")
                integrator = self.agents[integrator_id]
                model_type = "openai" if integrator["model"] and any(name in integrator["model"].lower() for name in ["gpt", "o1", "o3"]) else \
                            "anthropic" if integrator["model"] and any(name in integrator["model"].lower() for name in ["claude"]) else \
                            "openai"
                requestor = MMRequestor(
                    model_type=model_type,
                    model_name=integrator["model"],
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                self.requestors[integrator_id] = requestor
            
            response = await requestor.request(integration_prompt, temperature=0.3, max_tokens=2048)
            
            # Extract the response text
            response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else response
            
            # Update DBE interaction history for the integrator
            if integrator_id in self.dbe_instances:
                dbe_instance = self.dbe_instances[integrator_id]
                dbe_instance.update_interaction_history(integration_prompt, response_text)
                
                # Update boundary estimates for the integrator
                error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
                refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
                self.update_agent_boundaries(integrator_id, refined_estimates)
            
            return response_text
            
        except Exception as e:
            self._log(f"Error in solution synthesis: {str(e)}")
            # Fallback to a simple concatenation
            synthesis = "Synthesized solution (fallback due to error):\n\n"
            
            # Sort components by dependency order
            sorted_components = sorted(solution_components.items(), key=lambda x: x[0])
            
            for subtask_id, component in sorted_components:
                if isinstance(component, dict) and "result" in component:
                    synthesis += f"- {subtask_id}: {component['result']}\n"
                elif isinstance(component, str):
                    synthesis += f"- {subtask_id}: {component}\n"
            
            return synthesis
    
    async def validate_solution(self, solution, task):
        """
        Validate the synthesized solution
        
        Args:
            solution: Synthesized solution
            task: Original task
            
        Returns:
            Validated solution with confidence assessment
        """
        self._log("Validating final solution")
        
        # Select verifier agent
        verifier_id = self.select_verifier_agent()
        
        if not verifier_id:
            # Fallback to any available agent
            verifier_id = next(iter(self.agents.keys())) if self.agents else None
            
        if not verifier_id:
            self._log("No verifier agent available")
            return {
                "valid": True,  # Assume valid if no verifier available
                "confidence": 0.7,
                "feedback": "Solution could not be verified due to lack of verifier agent.",
                "verified_solution": solution
            }
        
        # Create the verification prompt
        verification_prompt = f"""
        You are a verification expert. Your task is to evaluate whether a solution correctly and completely addresses the original problem.
        
        Original problem:
        {task}
        
        Proposed solution:
        {solution}
        
        Please verify this solution by:
        1. Checking if all aspects of the problem are addressed
        2. Verifying the correctness of reasoning and calculations
        3. Identifying any errors, inconsistencies, or omissions
        
        Your response should include:
        1. A clear verdict on whether the solution is valid (Yes/No)
        2. Your confidence level in this assessment (0-100%)
        3. Specific feedback on any issues found
        4. A corrected or improved version if necessary
        
        Provide your verification:
        """
        
        try:
            # Get the verifier agent's response
            requestor = self.requestors.get(verifier_id)
            if not requestor:
                self._log(f"No requestor found for verifier agent {verifier_id}, creating a new one")
                verifier = self.agents[verifier_id]
                model_type = "openai" if verifier["model"] and any(name in verifier["model"].lower() for name in ["gpt", "o1", "o3"]) else \
                            "anthropic" if verifier["model"] and any(name in verifier["model"].lower() for name in ["claude"]) else \
                            "openai"
                requestor = MMRequestor(
                    model_type=model_type,
                    model_name=verifier["model"],
                    api_key=self.api_key,
                    api_base=self.api_base
                )
                self.requestors[verifier_id] = requestor
            
            response = await requestor.request(verification_prompt, temperature=0.3, max_tokens=2048)
            
            # Extract the response text
            response_text = response[-1]["content"][0]["text"] if isinstance(response, list) else response
            
            # Update DBE interaction history for the verifier
            if verifier_id in self.dbe_instances:
                dbe_instance = self.dbe_instances[verifier_id]
                dbe_instance.update_interaction_history(verification_prompt, response_text)
                
                # Update boundary estimates for the verifier
                error_patterns = dbe_instance.analyze_error_patterns(dbe_instance.interaction_history)
                refined_estimates = dbe_instance.refine_boundary_estimates(error_patterns)
                self.update_agent_boundaries(verifier_id, refined_estimates)
            
            # Parse the verification result
            valid = "yes" in response_text.lower() and "no" not in response_text.lower()[:50]
            
            # Extract confidence
            confidence_match = re.search(r'confidence.*?(\d+)%', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1))/100 if confidence_match else 0.7
            
            # Extract improved solution if provided
            improved_solution = solution
            if not valid and "corrected" in response_text.lower():
                # Try to extract improved solution
                improved_parts = response_text.split("corrected solution:", 1)
                if len(improved_parts) > 1:
                    improved_solution = improved_parts[1].strip()
                else:
                    # Try alternative formats
                    improved_parts = response_text.split("improved solution:", 1)
                    if len(improved_parts) > 1:
                        improved_solution = improved_parts[1].strip()
            
            return {
                "valid": valid,
                "confidence": confidence,
                "feedback": response_text,
                "verified_solution": improved_solution if not valid else solution
            }
            
        except Exception as e:
            self._log(f"Error in solution validation: {str(e)}")
            return {
                "valid": True,  # Assume valid if verification fails
                "confidence": 0.6,
                "feedback": f"Verification failed due to error: {str(e)}",
                "verified_solution": solution
            }
    
    async def solve_task(self, task):
        """
        Solve a complex reasoning task using the MARC framework
        
        Args:
            task: The reasoning task to solve
            
        Returns:
            Solved task with detailed collaboration record
        """
        # Initialize collaboration state
        self.task_assignments = {}
        self.solution_components = {}
        self.communication_history = []
        self.consensus_votes = defaultdict(dict)
        
        # Record the start of collaboration
        collaboration_record = {
            "task": task,
            "start_time": "simulated_timestamp",
            "agents": {agent_id: agent.copy() for agent_id, agent in self.agents.items()},
            "boundary_updates": {},
            "phases": []
        }
        
        self._log(f"Starting to solve task: {task}")
        
        # Initially deploy probes for all agents to get baseline boundary estimates
        for agent_id in self.agents:
            try:
                await self.deploy_dbe_probes(agent_id)
                # Record initial boundaries in collaboration record
                collaboration_record["boundary_updates"][agent_id] = {
                    "initial": self.agents[agent_id]["boundaries"].copy()
                }
            except Exception as e:
                self._log(f"Error deploying initial probes for agent {agent_id}: {e}")
        
        # Phase 1: Planning
        planner_id = self.select_planner_agent()
        if not planner_id:
            return {"error": "No suitable planner agent available."}
        
        # Decompose task into subtasks
        subtasks = await self.decompose_task(task, planner_id)
        
        collaboration_record["phases"].append({
            "phase": "planning",
            "planner_id": planner_id,
            "subtasks": subtasks
        })
        
        # Phase 2: Task Assignment
        task_assignments = self.assign_tasks(subtasks)
        self.task_assignments = task_assignments
        
        collaboration_record["phases"].append({
            "phase": "assignment",
            "assignments": task_assignments
        })
        
        # Phase 3: Collaborative Processing
        processing_rounds = []
        
        for round_num in range(self.max_communication_rounds):
            self._log(f"Processing round {round_num+1}/{self.max_communication_rounds}")
            round_record = {
                "round": round_num,
                "boundary_updates": {},
                "actions": []
            }
            
            # Process ready tasks
            ready_tasks = []
            for subtask_id, assignment in task_assignments.items():
                # Check if dependencies are satisfied
                subtask = assignment["subtask"]
                dependencies_met = True
                
                for dep_id in subtask.get("dependencies", []):
                    if dep_id not in self.solution_components:
                        dependencies_met = False
                        break
                
                if dependencies_met and subtask_id not in self.solution_components:
                    ready_tasks.append((subtask_id, assignment))
            
            if not ready_tasks:
                # No tasks ready or all tasks completed
                break
            
            # Record boundary updates for this round
            for agent_id in self.agents:
                round_record["boundary_updates"][agent_id] = self.agents[agent_id]["boundaries"].copy()
                
                # Also update the overall collaboration record
                if agent_id not in collaboration_record["boundary_updates"]:
                    collaboration_record["boundary_updates"][agent_id] = {}
                collaboration_record["boundary_updates"][agent_id][f"round_{round_num}"] = self.agents[agent_id]["boundaries"].copy()
            
            # Process each ready task
            for subtask_id, assignment in ready_tasks:
                agent_id = assignment["agent_id"]
                subtask = assignment["subtask"]
                
                self._log(f"Processing subtask {subtask_id} with agent {agent_id}")
                
                # Process the subtask
                result = await self.process_subtask(agent_id, subtask)
                
                action_record = {
                    "action": "process_subtask",
                    "agent_id": agent_id,
                    "subtask_id": subtask_id,
                    "result_status": result["status"],
                    "boundary_before": self.agents[agent_id]["boundaries"].copy()  # Capture boundaries before update
                }
                
                if result["status"] == "completed":
                    # Task completed successfully
                    self.solution_components[subtask_id] = result
                    
                    # Broadcast update to other agents
                    messages = self.broadcast_update(agent_id, subtask_id, result)
                    action_record["messages"] = len(messages)
                    
                    # Update agent status
                    self.agents[agent_id]["status"] = "idle"
                    self.agents[agent_id]["current_task"] = None
                    
                    # Update boundary estimates based on successful completion
                    if agent_id in self.dbe_instances:
                        dbe_instance = self.dbe_instances[agent_id]
                        # Successful completion might warrant a positive adjustment to boundaries
                        difficulty = self.estimate_task_difficulty(subtask)
                        for dim, diff in difficulty.items():
                            if dim in self.agents[agent_id]["boundaries"]:
                                # Slightly increase boundary for this dimension based on success
                                self.agents[agent_id]["boundaries"][dim] *= 1.05
                                self.agents[agent_id]["boundary_history"][dim].append(self.agents[agent_id]["boundaries"][dim])
                    
                    self._log(f"Subtask {subtask_id} completed successfully by agent {agent_id}")
                    
                elif result["status"] == "boundary_reached":
                    # Task exceeds agent's boundaries - request assistance
                    difficulty = self.estimate_task_difficulty(subtask)
                    assistance_plan = self.request_assistance(agent_id, subtask_id, difficulty)
                    
                    # Update boundary estimates to reflect the difficulty limits
                    if agent_id in self.dbe_instances:
                        for dim, diff in difficulty.items():
                            if dim in self.agents[agent_id]["boundaries"] and diff > self.agents[agent_id]["boundaries"][dim]:
                                # Adjust boundary downward based on failure
                                adjusted = max(0.9 * self.agents[agent_id]["boundaries"][dim], diff * 0.8)
                                self.agents[agent_id]["boundaries"][dim] = adjusted
                                self.agents[agent_id]["boundary_history"][dim].append(adjusted)
                    
                    action_record["assistance_plan"] = assistance_plan
                    action_record["boundary_after"] = self.agents[agent_id]["boundaries"].copy()  # Capture boundaries after update
                    
                    # If assistance available, process with collaboration
                    if assistance_plan:
                        # Process collaboratively
                        collaboration = assistance_plan[0]
                        lead_agent = collaboration["lead_agent"]
                        assistant_agent = collaboration["assistant_agent"]
                        focus_dimension = collaboration["focus_dimension"]
                        
                        self._log(f"Processing subtask {subtask_id} collaboratively between {lead_agent} and {assistant_agent}")
                        
                        collaborative_result = await self.process_collaborative_subtask(
                            lead_agent, assistant_agent, subtask_id, focus_dimension
                        )
                        
                        if collaborative_result["status"] == "completed":
                            self.solution_components[subtask_id] = collaborative_result
                            action_record["collaborative_result"] = "success"
                            
                            # Update boundaries based on successful collaboration
                            if lead_agent in self.dbe_instances and assistant_agent in self.dbe_instances:
                                # Lead agent gets a slight boundary decrease in the challenging dimension
                                if focus_dimension in self.agents[lead_agent]["boundaries"]:
                                    self.agents[lead_agent]["boundaries"][focus_dimension] *= 0.95
                                    self.agents[lead_agent]["boundary_history"][focus_dimension].append(
                                        self.agents[lead_agent]["boundaries"][focus_dimension]
                                    )
                                
                                # Assistant agent gets a slight boundary increase in their strong dimension
                                if focus_dimension in self.agents[assistant_agent]["boundaries"]:
                                    self.agents[assistant_agent]["boundaries"][focus_dimension] *= 1.05
                                    self.agents[assistant_agent]["boundary_history"][focus_dimension].append(
                                        self.agents[assistant_agent]["boundaries"][focus_dimension]
                                    )
                            
                            self._log(f"Collaborative processing of subtask {subtask_id} succeeded")
                        else:
                            # Collaborative approach failed, use consensus mechanism
                            action_record["collaborative_result"] = "failed"
                            self._log(f"Collaborative processing of subtask {subtask_id} failed, using consensus mechanism")
                            
                            # Move to consensus approach
                            options = [
                                "Simplify the approach and solve approximately",
                                "Break into smaller sub-components",
                                "Use external tools/resources",
                                "Provide partial solution with caveats"
                            ]
                            
                            votes = await self.reach_consensus(subtask_id, 
                                                         f"How to handle subtask {subtask_id} that exceeded boundaries?", 
                                                         options)
                            
                            action_record["consensus"] = votes
                            
                            # Apply consensus decision
                            if votes:
                                # Simulate resolution based on consensus
                                fallback_result = {
                                    "status": "completed_with_limitations",
                                    "subtask_id": subtask_id,
                                    "agent_id": agent_id,
                                    "result": f"Completed with limitations ({votes['option']}): {subtask['description']}",
                                    "confidence": 0.6,
                                    "message": f"Task completed using consensus approach: {votes['option']}"
                                }
                                
                                self.solution_components[subtask_id] = fallback_result
                                action_record["fallback_result"] = "applied_consensus"
                                
                                self._log(f"Applied consensus decision for subtask {subtask_id}: {votes['option']}")
                    else:
                        # No suitable assistance - use consensus mechanism
                        options = [
                            "Simplify the approach and solve approximately",
                            "Break into smaller sub-components",
                            "Use external tools/resources",
                            "Provide partial solution with caveats"
                        ]
                        
                        votes = await self.reach_consensus(subtask_id, 
                                                     f"How to handle subtask {subtask_id} that exceeded boundaries?", 
                                                     options)
                        
                        action_record["consensus"] = votes
                        
                        # Apply consensus decision
                        if votes:
                            # Simulate resolution based on consensus
                            fallback_result = {
                                "status": "completed_with_limitations",
                                "subtask_id": subtask_id,
                                "agent_id": agent_id,
                                "result": f"Completed with limitations ({votes['option']}): {subtask['description']}",
                                "confidence": 0.6,
                                "message": f"Task completed using consensus approach: {votes['option']}"
                            }
                            
                            self.solution_components[subtask_id] = fallback_result
                            action_record["fallback_result"] = "applied_consensus"
                            
                            self._log(f"Applied consensus decision for subtask {subtask_id}: {votes['option']}")
                
                round_record["actions"].append(action_record)
            
            processing_rounds.append(round_record)
            
            # Check if all subtasks are completed
            if len(self.solution_components) == len(subtasks):
                self._log("All subtasks completed")
                break
        
        collaboration_record["phases"].append({
            "phase": "processing",
            "rounds": processing_rounds
        })
        
        # Phase 4: Integration
        integrator_id = self.select_integrator_agent()
        
        if not integrator_id:
            integrator_id = planner_id  # Fallback to planner
            self._log(f"No dedicated integrator available, using planner {planner_id} instead")
        
        # Synthesize solution
        synthesized_solution = await self.synthesize_solution(self.solution_components)
        
        collaboration_record["phases"].append({
            "phase": "integration",
            "integrator_id": integrator_id,
            "solution_length": len(synthesized_solution)
        })
        
        # Phase 5: Verification
        verifier_id = self.select_verifier_agent()
        
        if not verifier_id:
            verifier_id = integrator_id  # Fallback to integrator
            self._log(f"No dedicated verifier available, using integrator {integrator_id} instead")
        
        # Validate solution
        validation_result = await self.validate_solution(synthesized_solution, task)
        
        collaboration_record["phases"].append({
            "phase": "verification",
            "verifier_id": verifier_id,
            "validation_result": validation_result["valid"],
            "confidence": validation_result["confidence"]
        })
        
        # Final result
        collaboration_record["final_solution"] = validation_result["verified_solution"]
        collaboration_record["completion_status"] = "success" if validation_result["valid"] else "partial"
        collaboration_record["end_time"] = "simulated_timestamp"
        collaboration_record["messages_exchanged"] = len(self.communication_history)
        
        # Final boundary states
        collaboration_record["final_boundaries"] = {agent_id: agent["boundaries"].copy() for agent_id, agent in self.agents.items()}
        collaboration_record["boundary_history"] = {agent_id: agent["boundary_history"] for agent_id, agent in self.agents.items()}
        
        self._log("Task solving completed")
        return collaboration_record
