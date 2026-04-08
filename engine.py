import random
from typing import List, Tuple, Dict, Any
from models import CloudResource, Observation, Action, Reward, ResourceType, ActionType

class CloudSweepEngine:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.resources: List[CloudResource] = self._generate_initial_resources()
        self.day = 0
        self.total_saved = 0.0
        self.dangerous_actions = 0

    def _generate_initial_resources(self) -> List[CloudResource]:
        """Generate 10-20 mock cloud resources"""
        resources = []
        num_resources = random.randint(10, 20)

        for i in range(num_resources):
            resource_id = f"{random.choice(['EC2', 'S3', 'RDS'])}-{i:03d}"
            resource_type = ResourceType(random.choice(['EC2', 'S3', 'RDS']))

            # Cost per day varies by type
            if resource_type == ResourceType.EC2:
                cost_per_day = random.uniform(5.0, 50.0)
                cpu_usage_avg = random.uniform(0.0, 100.0)
            elif resource_type == ResourceType.S3:
                cost_per_day = random.uniform(0.1, 10.0)
                cpu_usage_avg = 0.0  # S3 doesn't have CPU
            else:  # RDS
                cost_per_day = random.uniform(10.0, 100.0)
                cpu_usage_avg = random.uniform(0.0, 100.0)

            is_production = random.choice([True, False])
            # Production resources are less likely to be unattended
            if is_production:
                last_accessed_days = random.randint(0, 15)
            else:
                last_accessed_days = random.randint(0, 60)

            resources.append(CloudResource(
                id=resource_id,
                type=resource_type,
                cost_per_day=cost_per_day,
                cpu_usage_avg=cpu_usage_avg,
                is_production=is_production,
                last_accessed_days=last_accessed_days
            ))

        return resources

    def _calculate_total_daily_cost(self) -> float:
        return sum(resource.cost_per_day for resource in self.resources)

    def _get_observation(self) -> Observation:
        return Observation(
            resources=self.resources,
            total_daily_cost=self._calculate_total_daily_cost(),
            day=self.day
        )

    def reset(self) -> Observation:
        """Reset the environment to initial state"""
        self.__init__(seed=random.randint(0, 10000))
        return self._get_observation()

    def state(self) -> Observation:
        """Return current state of the environment"""
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        reward_components = {}
        reward_value = 0.0
        info = {"action_taken": action.action_type.value, "resource_id": action.resource_id}

        # Find the resource
        resource = None
        for res in self.resources:
            if res.id == action.resource_id:
                resource = res
                break

        if resource is None:
            # Invalid resource ID
            reward_value = -0.1
            reward_components["invalid_resource"] = -0.1
            done = False
            return self._get_observation(), reward_value, done, info

        # Process the action
        if action.action_type == ActionType.DELETE:
            if resource.is_production:
                # Dangerous action: deleting production resource
                reward_value = -0.5
                self.dangerous_actions += 1
                reward_components["dangerous_delete"] = -0.5
                info["dangerous"] = True
            else:
                # Safe deletion: remove resource and add savings
                saved_cost = resource.cost_per_day
                self.total_saved += saved_cost
                reward_value = min(0.1 * (saved_cost / 10.0), 0.5)  # Cap at 0.5 per step
                reward_components["cost_savings"] = reward_value
                self.resources.remove(resource)
                info["saved_cost"] = saved_cost

        elif action.action_type == ActionType.RIGHT_SIZE:
            # Only applicable to EC2 and RDS (have CPU)
            if resource.type in [ResourceType.EC2, ResourceType.RDS]:
                # Right-size: reduce cost by 30% if CPU < 5% for 7 days
                if resource.cpu_usage_avg < 5.0 and resource.last_accessed_days > 7:
                    saved_cost = resource.cost_per_day * 0.3
                    resource.cost_per_day *= 0.7
                    self.total_saved += saved_cost
                    reward_value = min(0.1 * (saved_cost / 10.0), 0.5)
                    reward_components["right_size_savings"] = reward_value
                    info["saved_cost"] = saved_cost
                else:
                    # Attempted right-size but conditions not met
                    reward_value = -0.05
                    reward_components["ineffective_right_size"] = -0.05
            else:
                # Cannot right-size S3
                reward_value = -0.05
                reward_components["invalid_action_for_type"] = -0.05

        elif action.action_type == ActionType.TAG_FOR_REVIEW:
            # Tagging production idle resources
            if resource.is_production and resource.cpu_usage_avg < 5.0:
                reward_value = 0.05  # Small reward for correct identification
                reward_components["tagging_reward"] = 0.05
                info["tagged_for_review"] = True
            else:
                reward_value = -0.02
                reward_components["incorrect_tagging"] = -0.02

        elif action.action_type == ActionType.NOTIFY_OWNER:
            # Similar to tagging but for notification
            if resource.is_production and resource.cpu_usage_avg < 5.0:
                reward_value = 0.03
                reward_components["notification_reward"] = 0.03
                info["notified_owner"] = True
            else:
                reward_value = -0.02
                reward_components["incorrect_notification"] = -0.02

        elif action.action_type == ActionType.NO_OP:
            reward_value = 0.0
            reward_components["no_op"] = 0.0

        # Ensure reward is between 0 and 1 as per spec
        # According to spec, reward should be between 0.0 and 1.0
        final_reward = max(0.0, min(1.0, reward_value))

        # Increment day
        self.day += 1

        # Check if episode is done (e.g., after 30 days or no resources left)
        done = self.day >= 30 or len(self.resources) == 0

        # Update info
        info.update({
            "total_saved": self.total_saved,
            "dangerous_actions": self.dangerous_actions,
            "remaining_resources": len(self.resources),
            "reward_components": reward_components
        })

        return self._get_observation(), final_reward, done, info

# For testing
if __name__ == "__main__":
    engine = CloudSweepEngine()
    obs = engine.reset()
    print(f"Initial observation: {len(obs.resources)} resources, total daily cost: {obs.total_daily_cost:.2f}")

    # Example action: delete first non-production unattached resource
    for resource in obs.resources:
        if not resource.is_production and resource.last_accessed_days > 30:
            action = Action(resource_id=resource.id, action_type=ActionType.DELETE)
            next_obs, reward, done, info = engine.step(action)
            print(f"Action: DELETE {resource.id}, Reward: {reward}, Done: {done}")
            print(f"Info: {info}")
            break