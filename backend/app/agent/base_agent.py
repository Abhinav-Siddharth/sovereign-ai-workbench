"""Agent with planning, tool use, observation, and critique."""

from typing import Any, Dict

from backend.app.models.base import BaseModel
from backend.app.agent.tools import RAGSearchTool


class Agent:
    """Sovereign agent that can plan and use local tools."""

    def __init__(self, model: BaseModel) -> None:
        """Initialize the agent."""
        self.model = model
        self.rag_tool = RAGSearchTool()

    def plan(self, task: str) -> str:
        """Create a plan for the task."""

        prompt = (
            "You are an AI planning agent.\n"
            "Create a clear, concise step-by-step plan "
            "to accomplish the following task.\n\n"
            f"Task: {task}\n\n"
            "Plan:"
        )

        return self.model.generate(prompt)

    def act(self, task: str, plan: str) -> Dict[str, Any]:
        """Execute the plan using available tools."""

        # Search the local knowledge base for relevant information.
        rag_result = self.rag_tool.run(
            query=task,
            n_results=3,
        )

        return {
            "tool": self.rag_tool.name,
            "tool_input": task,
            "tool_result": rag_result,
            "plan": plan,
        }

    def observe(self, result: Dict[str, Any]) -> str:
        """Observe and summarize the tool execution result."""

        prompt = (
            "You are an AI observer.\n"
            "Inspect the following tool execution result and "
            "describe what was found.\n\n"
            f"Tool execution result:\n{result}\n\n"
            "Observations:"
        )

        return self.model.generate(prompt)

    def critique(
        self,
        task: str,
        result: Dict[str, Any],
    ) -> str:
        """Evaluate whether the result satisfies the task."""

        prompt = (
            "You are an AI critique agent.\n"
            "Evaluate whether the following result accurately "
            "and completely satisfies the original task.\n\n"
            f"Original Task: {task}\n\n"
            f"Execution Result:\n{result}\n\n"
            "Critique:"
        )

        return self.model.generate(prompt)

    def run(self, task: str) -> Dict[str, Any]:
        """Run the complete agent workflow."""

        # 1. Plan
        plan_output = self.plan(task)

        # 2. Act using a real tool
        execution_output = self.act(
            task=task,
            plan=plan_output,
        )

        # 3. Observe the result
        observation_output = self.observe(
            execution_output
        )

        # 4. Critique the result
        critique_output = self.critique(
            task=task,
            result=execution_output,
        )

        # Return the complete agent trace.
        return {
            "task": task,
            "plan": plan_output,
            "execution": execution_output,
            "observation": observation_output,
            "critique": critique_output,
        }