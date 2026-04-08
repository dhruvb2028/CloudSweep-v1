#!/usr/bin/env python3
"""
Baseline inference script for CloudSweep-v1 environment.
Uses OpenAI client to generate actions and emits OpenEnv-compliant log formats.
"""

import os
import sys
import json
from typing import Dict, Any
from openai import OpenAI
from engine import CloudSweepEngine
from models import Observation, Action, ActionType, ResourceType

def log_start(episode_id: int, task_description: str):
    """Emit [START] log format"""
    print(f"[START] episode_id:{episode_id} task:{task_description}")

def log_step(step_num: int, action: Action, reward: float, total_reward: float, observation: Observation):
    """Emit [STEP] log format"""
    action_desc = f"{action.action_type.value}:{action.resource_id}"
    print(f"[STEP] step:{step_num} action:{action_desc} reward:{reward:.3f} total_reward:{total_reward:.3f}")

def log_end(episode_id: int, total_reward: float, length: int, success: bool):
    """Emit [END] log format"""
    print(f"[END] episode_id:{episode_id} total_reward:{total_reward:.3f} length:{length} success:{success}")

def select_action(observation: Observation, task: str) -> Action:
    """
    Select an action based on the current observation and task using OpenAI API.
    Falls back to rule-based policy if API key is not available.
    """
    # Try to use OpenAI API if key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)

            # Prepare observation summary for the prompt
            resources_summary = []
            for resource in observation.resources:
                resources_summary.append(
                    f"- {resource.id}: {resource.type.value}, "
                    f"${resource.cost_per_day:.2f}/day, "
                    f"{'Production' if resource.is_production else 'Non-production'}, "
                    f"CPU: {resource.cpu_usage_avg:.1f}%, "
                    f"Last accessed: {resource.last_accessed_days} days ago"
                )

            prompt = f"""You are a cloud cost optimization agent. Your task is: {task}

Current cloud resources:
{chr(10).join(resources_summary)}

Available actions:
- delete: Remove a resource (only safe for non-production resources unattached >30 days)
- right_size: Reduce instance size by 30% (only for EC2/RDS with CPU <5% for 7+ days)
- tag_for_review: Tag production idle resources (CPU <5%) for review
- notify_owner: Notify owner of production idle resources (CPU <5%)
- no_op: Take no action

Choose the best action to take. Respond with JSON format:
{{"action_type": "action_name", "resource_id": "resource_id"}}

If no action is appropriate, use no_op with any resource ID."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cloud cost optimization expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )

            import json
            result = json.loads(response.choices[0].message.content)
            action_type = result["action_type"]
            resource_id = result["resource_id"]

            # Validate action_type
            try:
                action_enum = ActionType(action_type)
                return Action(resource_id=resource_id, action_type=action_enum)
            except ValueError:
                # Fallback to rule-based if invalid action type
                pass

        except Exception as e:
            # If OpenAI API fails, fall back to rule-based
            pass

    # Fallback rule-based policy
    # Simple heuristic: look for unattached non-production resources to delete (Task 1)
    if "unattached" in task.lower():
        for resource in observation.resources:
            if not resource.is_production and resource.last_accessed_days > 30:
                return Action(resource_id=resource.id, action_type=ActionType.DELETE)

    # Look for over-provisioned instances to right-size (Task 2)
    elif "right-size" in task.lower() or "overprovisioned" in task.lower():
        for resource in observation.resources:
            if resource.type in [ResourceType.EC2, ResourceType.RDS]:
                # More flexible criteria for demonstration
                if resource.cpu_usage_avg < 10.0 and resource.last_accessed_days > 3:
                    return Action(resource_id=resource.id, action_type=ActionType.RIGHT_SIZE)

    # Handle production idle resources (Task 3)
    elif "production" in task.lower() and "idle" in task.lower():
        for resource in observation.resources:
            if resource.is_production and resource.cpu_usage_avg < 5.0:
                # Tag for review as safer first step
                return Action(resource_id=resource.id, action_type=ActionType.TAG_FOR_REVIEW)

    # Default: no-op
    return Action(resource_id=observation.resources[0].id if observation.resources else "dummy",
                  action_type=ActionType.NO_OP)

def run_episode(task_description: str, max_steps: int = 50) -> Dict[str, Any]:
    """Run a single episode in the CloudSweep-v1 environment"""
    # Initialize environment
    env = CloudSweepEngine(seed=42)
    observation = env.reset()

    total_reward = 0.0
    episode_id = hash(str(observation)) % 10000  # Simple episode ID

    # Log start
    log_start(episode_id, task_description)

    # Run episode
    for step_num in range(1, max_steps + 1):
        # Select action (in practice, this would call OpenAI API)
        action = select_action(observation, task_description)

        # Execute step
        observation, reward, done, info = env.step(action)
        total_reward += reward

        # Log step
        log_step(step_num, action, reward, total_reward, observation)

        if done:
            break

    # Determine success based on task-specific criteria (simplified)
    success = total_reward > 0  # Simplified success criterion

    # Log end
    log_end(episode_id, total_reward, step_num, success)

    return {
        "episode_id": episode_id,
        "total_reward": total_reward,
        "length": step_num,
        "success": success
    }

def main():
    """Main function to run inference"""
    # Get task from environment or use default
    task_description = os.getenv("TASK_DESCRIPTION", "General cloud cost optimization")

    # Run episode
    result = run_episode(task_description)

    # Print summary to stderr (keep stdout clean for logging)
    print(f"Episode completed: {json.dumps(result, indent=2)}", file=sys.stderr)

if __name__ == "__main__":
    main()