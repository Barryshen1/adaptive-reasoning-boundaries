"""
Implementation of Multi-Agent Reasoning Collaboration (MARC)
"""
import numpy as np
import json
import re
from collections import defaultdict


class MARC:
    """
    Multi-Agent Reasoning Collaboration (MARC)
    
    MARC leverages multiple specialized agents with complementary reasoning 
    boundaries to collaboratively solve complex tasks.
    """
    
    def __init__(self, 
                 agents=None,               # Dictionary of available agents with their boundary profiles
                 max_communication_rounds=5  # Maximum number of communication rounds
                 ):
        """
        Initialize MARC with agent configuration
        
        Args:
            agents: Dictionary of available agents with their boundary profiles
            max_communication_rounds: Maximum number of communication rounds
        """
        self.agents = agents or {}
        self.max_communication_rounds = max_communication_rounds
        
        # Initialize collaboration state
        self.task_assignments = {}  # Map of subtasks to agents
        self.solution_components = {}  # Partial solutions by subtask
        self.communication_history = []  # Record of agent communications
        self.consensus_votes = defaultdict(dict)  # Votes on critical decisions
        self.subtasks = []  # List of all subtasks
    
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
            "status": "idle"
        }
        
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
                
        return best_integrator
    
    def decompose_task(self, task, planner_agent_id):
        """
        Decompose the main task into subtasks using the planner agent
        
        Args:
            task: The main reasoning task
            planner_agent_id: Agent ID of the selected planner
            
        Returns:
            List of subtasks
        """
        # In a real implementation, this would call the planner agent's LLM
        # For this template, we'll provide a simulated decomposition
        
        # Extract task type for appropriate decomposition
        task_type = self._identify_task_type(task)
        
        if task_type == "mathematical":
            subtasks = self._decompose_mathematical_task(task)
        elif task_type == "logical":
            subtasks = self._decompose_logical_task(task)
        else:
            subtasks = self._decompose_general_task(task)
            
        # Save subtasks for future reference
        self.subtasks = subtasks
        
        return subtasks
    
    def _identify_task_type(self, task):
        """
        Identify the type of reasoning task
        
        Args:
            task: The reasoning task
            
        Returns:
            Task type identifier
        """
        # In a real implementation, this would use NLP techniques
        # For this template, we'll use a simple keyword-based approach
        
        if any(keyword in task.lower() for keyword in ["calculate", "compute", "math", "equation", "solve", "arithmetic"]):
            return "mathematical"
        elif any(keyword in task.lower() for keyword in ["logic", "deduce", "infer", "prove", "conclude", "syllogism"]):
            return "logical"
        else:
            return "general"
    
    def _decompose_mathematical_task(self, task):
        """
        Decompose a mathematical task into subtasks
        
        Args:
            task: The mathematical reasoning task
            
        Returns:
            List of subtasks
        """
        # This is a simplified decomposition - would be more sophisticated in practice
        subtasks = [
            {"id": "parse", "description": f"Parse the mathematical problem: {task}", "dependencies": []},
            {"id": "formulate", "description": "Formulate the mathematical equations needed", "dependencies": ["parse"]},
            {"id": "solve", "description": "Solve the equations step by step", "dependencies": ["formulate"]},
            {"id": "verify", "description": "Verify the solution by checking steps and substituting back", "dependencies": ["solve"]},
            {"id": "explain", "description": "Explain the final answer in context of the original problem", "dependencies": ["verify"]}
        ]
        
        return subtasks
    
    def _decompose_logical_task(self, task):
        """
        Decompose a logical reasoning task into subtasks
        
        Args:
            task: The logical reasoning task
            
        Returns:
            List of subtasks
        """
        subtasks = [
            {"id": "premises", "description": f"Identify the key premises in the problem: {task}", "dependencies": []},
            {"id": "rules", "description": "Identify logical rules or principles that apply", "dependencies": ["premises"]},
            {"id": "infer", "description": "Draw logical inferences step by step", "dependencies": ["premises", "rules"]},
            {"id": "validate", "description": "Check for logical fallacies or contradictions", "dependencies": ["infer"]},
            {"id": "conclude", "description": "Form a final conclusion based on the validated reasoning", "dependencies": ["validate"]}
        ]
        
        return subtasks
    
    def _decompose_general_task(self, task):
        """
        Decompose a general reasoning task into subtasks
        
        Args:
            task: The general reasoning task
            
        Returns:
            List of subtasks
        """
        subtasks = [
            {"id": "analyze", "description": f"Analyze the key components of the problem: {task}", "dependencies": []},
            {"id": "research", "description": "Identify relevant facts and context", "dependencies": ["analyze"]},
            {"id": "structure", "description": "Structure an approach to address the problem", "dependencies": ["analyze", "research"]},
            {"id": "reason", "description": "Apply reasoning to draw conclusions", "dependencies": ["structure"]},
            {"id": "summarize", "description": "Summarize findings and provide a final answer", "dependencies": ["reason"]}
        ]
        
        return subtasks
    
    def estimate_task_difficulty(self, subtask):
        """
        Estimate difficulty vector for a subtask
        
        Args:
            subtask: The subtask to assess
            
        Returns:
            Difficulty vector across reasoning dimensions
        """
        # In a real implementation, this would use NLP to analyze the task
        # For this template, we'll simulate difficulty estimation
        
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
        
        return task_assignments
    
    def process_subtask(self, agent_id, subtask):
        """
        Process a subtask with a specific agent
        
        Args:
            agent_id: Agent ID assigned to the task
            subtask: The subtask to process
            
        Returns:
            Subtask result and status
        """
        # In a real implementation, this would call the agent's LLM
        # For this template, we'll simulate processing
        
        difficulty = self.estimate_task_difficulty(subtask)
        agent = self.agents[agent_id]
        
        # Check if task is beyond agent's boundaries
        beyond_boundary = False
        for dim, diff in difficulty.items():
            if dim in agent["boundaries"] and diff > agent["boundaries"][dim] * 1.2:
                beyond_boundary = True
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
        
        # Simulate successful processing
        # In reality, this would contain the actual reasoning output
        return {
            "status": "completed",
            "subtask_id": subtask["id"],
            "agent_id": agent_id,
            "result": f"Completed reasoning for {subtask['description']}",
            "confidence": np.random.uniform(0.7, 0.95),
            "message": "Task completed successfully."
        }
    
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
        
        return assistance_plan
    
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
    
    def _get_vote_probabilities(self, agent_id, options):
        """
        Get probability distribution for agent's vote
        
        Args:
            agent_id: Agent ID
            options: List of options
            
        Returns:
            Probability distribution
        """
        # In a real implementation, this would be based on agent's reasoning
        # For this template, simulate a skewed distribution
        n_options = len(options)
        agent_type = self.agents.get(agent_id, {}).get("type", "")
        
        if agent_type == "calculator":
            # Calculator prefers systematic approaches
            probs = np.array([0.1, 0.4, 0.2, 0.3])
        elif agent_type == "planner":
            # Planner prefers structured approaches
            probs = np.array([0.2, 0.1, 0.5, 0.2])
        elif agent_type == "verifier":
            # Verifier prefers conservative approaches
            probs = np.array([0.3, 0.2, 0.1, 0.4])
        else:
            # Default uniform distribution
            probs = np.ones(4) / 4
        
        # Pad or truncate to match number of options
        if len(probs) < n_options:
            probs = np.pad(probs, (0, n_options - len(probs)), 'constant', constant_values=(0.1,))
        elif len(probs) > n_options:
            probs = probs[:n_options]
        
        # Normalize
        return probs / probs.sum()
    
    def reach_consensus(self, subtask_id, question, options):
        """
        Reach consensus through weighted voting
        
        Args:
            subtask_id: Subtask ID the consensus relates to
            question: The decision question
            options: Possible options
            
        Returns:
            Consensus decision
        """
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
                
                # Get agent's vote on the question
                # In a real implementation, this would call the agent's model
                # For this template, simulate a vote
                preference_idx = np.random.choice(len(options), p=self._get_vote_probabilities(agent_id, options))
                preference = options[preference_idx]
                
                votes[agent_id] = {
                    "option": preference,
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
    
    def synthesize_solution(self, solution_components):
        """
        Synthesize final solution from component parts
        
        Args:
            solution_components: Dictionary of partial solutions by subtask
            
        Returns:
            Synthesized complete solution
        """
        # In a real implementation, this would use the integrator agent's LLM
        # For this template, we'll simulate synthesis
        
        if not solution_components:
            return "No solution components available."
        
        # Placeholder for synthesized solution
        synthesis = "Synthesized solution based on collaborative reasoning:\n\n"
        
        # Sort components by dependency order
        sorted_components = sorted(solution_components.items(), key=lambda x: x[0])
        
        for subtask_id, component in sorted_components:
            if isinstance(component, dict) and "result" in component:
                synthesis += f"- {subtask_id}: {component['result']}\n"
            elif isinstance(component, str):
                synthesis += f"- {subtask_id}: {component}\n"
        
        synthesis += "\nFinal answer: "
        # In a real implementation, this would generate a coherent final answer
        synthesis += "Based on the collaborative analysis above, the answer is [ANSWER]."
        
        return synthesis
    
    def validate_solution(self, solution, task):
        """
        Validate the synthesized solution
        
        Args:
            solution: Synthesized solution
            task: Original task
            
        Returns:
            Validated solution with confidence assessment
        """
        # In a real implementation, this would use the verifier agent's LLM
        # For this template, we'll simulate validation
        
        # Simulate validation checks
        validation_result = {
            "valid": True,
            "confidence": 0.85,
            "feedback": "Solution validated successfully. All reasoning steps are sound and the final answer follows logically from the steps.",
            "verified_solution": solution
        }
        
        return validation_result
    
    def solve_task(self, task):
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
            "phases": []
        }
        
        # Phase 1: Planning
        planner_id = self.select_planner_agent()
        if not planner_id:
            return {"error": "No suitable planner agent available."}
        
        # Decompose task into subtasks
        subtasks = self.decompose_task(task, planner_id)
        
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
            round_record = {
                "round": round_num,
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
            
            # Process each ready task
            for subtask_id, assignment in ready_tasks:
                agent_id = assignment["agent_id"]
                subtask = assignment["subtask"]
                
                # Process the subtask
                result = self.process_subtask(agent_id, subtask)
                
                action_record = {
                    "action": "process_subtask",
                    "agent_id": agent_id,
                    "subtask_id": subtask_id,
                    "result_status": result["status"]
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
                    
                elif result["status"] == "boundary_reached":
                    # Task exceeds agent's boundaries - request assistance
                    difficulty = self.estimate_task_difficulty(subtask)
                    assistance_plan = self.request_assistance(agent_id, subtask_id, difficulty)
                    
                    action_record["assistance_plan"] = assistance_plan
                    
                    # If assistance available, process with collaboration
                    if assistance_plan:
                        # Simulate collaborative processing
                        collaborative_result = {
                            "status": "completed",
                            "subtask_id": subtask_id,
                            "agent_id": f"{agent_id}+{assistance_plan[0]['assistant_agent']}",
                            "result": f"Collaboratively completed reasoning for {subtask['description']}",
                            "confidence": 0.8,
                            "message": "Task completed through collaboration."
                        }
                        
                        self.solution_components[subtask_id] = collaborative_result
                        action_record["collaborative_result"] = "success"
                    else:
                        # No suitable assistance - use consensus mechanism
                        options = [
                            "Simplify the approach and solve approximately",
                            "Break into smaller sub-components",
                            "Use external tools/resources",
                            "Provide partial solution with caveats"
                        ]
                        
                        votes = self.reach_consensus(subtask_id, 
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
                
                round_record["actions"].append(action_record)
            
            processing_rounds.append(round_record)
            
            # Check if all subtasks are completed
            if len(self.solution_components) == len(subtasks):
                break
        
        collaboration_record["phases"].append({
            "phase": "processing",
            "rounds": processing_rounds
        })
        
        # Phase 4: Integration
        integrator_id = self.select_integrator_agent()
        
        if not integrator_id:
            integrator_id = planner_id  # Fallback to planner
        
        # Synthesize solution
        synthesized_solution = self.synthesize_solution(self.solution_components)
        
        collaboration_record["phases"].append({
            "phase": "integration",
            "integrator_id": integrator_id,
            "solution_length": len(synthesized_solution)
        })
        
        # Phase 5: Verification
        verifier_id = self.select_verifier_agent()
        
        if not verifier_id:
            verifier_id = integrator_id  # Fallback to integrator
        
        # Validate solution
        validation_result = self.validate_solution(synthesized_solution, task)
        
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
        
        return collaboration_record
