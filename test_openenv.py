#!/usr/bin/env python3
"""
Test script to validate OpenEnv compliance of CloudSweep-v1 environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine import CloudSweepEngine
from models import Observation, Action, ActionType

def test_reset():
    """Test reset method returns initial observation"""
    print("Testing reset()...")
    env = CloudSweepEngine(seed=42)
    obs = env.reset()

    assert isinstance(obs, Observation), "reset() should return Observation"
    assert len(obs.resources) > 0, "Should have resources"
    assert obs.day == 0, "Day should start at 0"
    assert obs.total_daily_cost > 0, "Should have positive total cost"
    print("[PASS] reset() test passed")

def test_state():
    """Test state method returns current state"""
    print("Testing state()...")
    env = CloudSweepEngine(seed=42)
    obs = env.reset()
    state = env.state()

    assert isinstance(state, Observation), "state() should return Observation"
    assert len(state.resources) == len(obs.resources), "State should match observation"
    assert state.day == obs.day, "State day should match observation day"
    assert state.total_daily_cost == obs.total_daily_cost, "State cost should match observation cost"
    print("[PASS] state() test passed")

def test_step():
    """Test step method executes action and returns proper tuple"""
    print("Testing step()...")
    env = CloudSweepEngine(seed=42)
    obs = env.reset()

    # Take a safe action - delete a non-production unattached resource
    action_taken = None
    for resource in obs.resources:
        if not resource.is_production and resource.last_accessed_days > 30:
            action = Action(resource_id=resource.id, action_type=ActionType.DELETE)
            action_taken = action
            break

    if action_taken is None:
        # Fallback to no-op if no suitable resource found
        action = Action(resource_id=obs.resources[0].id, action_type=ActionType.NO_OP)
        action_taken = action

    next_obs, reward, done, info = env.step(action)

    # Check return types
    assert isinstance(next_obs, Observation), "step() observation should be Observation"
    assert isinstance(reward, float), "step() reward should be float"
    assert isinstance(done, bool), "step() done should be bool"
    assert isinstance(info, dict), "step() info should be dict"

    # Check info contains expected fields
    assert "action_taken" in info, "info should contain action_taken"
    assert "resource_id" in info, "info should contain resource_id"

    print("[PASS] step() test passed")

def test_action_types():
    """Test that all action types can be processed"""
    print("Testing action types...")
    env = CloudSweepEngine(seed=42)
    obs = env.reset()

    action_types = [
        ActionType.DELETE,
        ActionType.RIGHT_SIZE,
        ActionType.TAG_FOR_REVIEW,
        ActionType.NOTIFY_OWNER,
        ActionType.NO_OP
    ]

    for action_type in action_types:
        # Use first resource for testing
        action = Action(resource_id=obs.resources[0].id, action_type=action_type)
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, float), f"Reward should be float for {action_type}"
        print(f"  [PASS] {action_type.value} action processed")

    print("[PASS] action types test passed")

def test_reward_bounds():
    """Test that reward is clipped to [0, 1] range"""
    print("Testing reward bounds...")
    env = CloudSweepEngine(seed=42)
    obs = env.reset()

    # Test dangerous action (should give negative reward but clipped to 0)
    for resource in obs.resources:
        if resource.is_production:
            action = Action(resource_id=resource.id, action_type=ActionType.DELETE)
            obs, reward, done, info = env.step(action)
            assert reward >= 0.0 and reward <= 1.0, f"Reward {reward} should be in [0,1]"
            break

    print("[PASS] reward bounds test passed")

def main():
    """Run all tests"""
    print("Running OpenEnv compliance tests for CloudSweep-v1...\n")

    try:
        test_reset()
        test_state()
        test_step()
        test_action_types()
        test_reward_bounds()

        print("\n[SUCCESS] All tests passed! Environment is OpenEnv compliant.")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)