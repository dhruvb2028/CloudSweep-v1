#!/usr/bin/env python3
"""Simple test script for CloudSweep-v1 environment"""

import requests
import json

def test_health():
    print("=== Testing Health Endpoint ===")
    r = requests.get('http://localhost:8000/health')
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
    assert r.status_code == 200
    print("Health check passed\n")

def test_reset():
    print("=== Testing Reset Endpoint ===")
    r = requests.post('http://localhost:8000/reset')
    print(f"Status: {r.status_code}")
    assert r.status_code == 200
    obs = r.json()
    print(f"Resources: {len(obs['resources'])}")
    print(f"Total daily cost: ${obs['total_daily_cost']:.2f}")
    print(f"Day: {obs['day']}")

    # Show first resource
    if obs['resources']:
        res = obs['resources'][0]
        print(f"Sample resource: {res['id']} ({res['type']})")
        print(f"  Cost: ${res['cost_per_day']:.2f}/day")
        print(f"  CPU: {res['cpu_usage_avg']:.1f}%")
        print(f"  Production: {res['is_production']}")
        print(f"  Last accessed: {res['last_accessed_days']} days ago")
    print("Reset passed\n")
    return obs

def test_step(action_dict, description):
    print(f"=== Testing {description} ===")
    r = requests.post('http://localhost:8000/step', json={'action': action_dict})
    print(f"Status: {r.status_code}")
    assert r.status_code == 200
    result = r.json()
    print(f"Reward: {result['reward']:.3f}")
    print(f"Done: {result['done']}")
    print(f"Info keys: {list(result['info'].keys())}")

    # Show relevant info
    if 'saved_cost' in result['info']:
        print(f"Saved cost: ${result['info']['saved_cost']:.2f}")
    if 'dangerous' in result['info']:
        print(f"Dangerous action: {result['info']['dangerous']}")
    if 'tagged_for_review' in result['info']:
        print(f"Tagged for review: {result['info']['tagged_for_review']}")
    print("Step test passed\n")
    return result

def analyze_resources(obs):
    print("=== Resource Analysis ===")
    resources = obs['resources']

    # Count by type
    types = {}
    for res in resources:
        t = res['type']
        types[t] = types.get(t, 0) + 1
    print(f"Resource types: {types}")

    # Count production
    prod_count = sum(1 for r in resources if r['is_production'])
    print(f"Production resources: {prod_count}/{len(resources)}")

    # Task 1 targets: unattached non-production
    task1 = [r for r in resources if not r['is_production'] and r['last_accessed_days'] > 30]
    print(f"Task 1 targets (unattached non-production): {len(task1)}")
    if task1:
        savings = sum(r['cost_per_day'] for r in task1)
        print(f"  Potential daily savings: ${savings:.2f}")
        print(f"  Example: {task1[0]['id']} (${task1[0]['cost_per_day']:.2f}/day, {task1[0]['last_accessed_days']} days)")

    # Task 2 targets: over-provisioned instances
    task2 = [r for r in resources if r['type'] in ['EC2', 'RDS'] and r['cpu_usage_avg'] < 5.0 and r['last_accessed_days'] > 7]
    print(f"Task 2 targets (over-provisioned EC2/RDS): {len(task2)}")
    if task2:
        savings = sum(r['cost_per_day'] * 0.3 for r in task2)  # 30% reduction
        print(f"  Potential daily savings: ${savings:.2f}")
        print(f"  Example: {task2[0]['id']} (CPU: {task2[0]['cpu_usage_avg']:.1f}%, {task2[0]['last_accessed_days']} days)")

    # Task 3 targets: production idle
    task3 = [r for r in resources if r['is_production'] and r['cpu_usage_avg'] < 5.0]
    print(f"Task 3 targets (production idle): {len(task3)}")
    if task3:
        print(f"  Example: {task3[0]['id']} (CPU: {task3[0]['cpu_usage_avg']:.1f}%, Production)")

    print()

def main():
    print("CloudSweep-v1 Environment Test Suite")
    print("=" * 50)

    # Test health
    test_health()

    # Test reset and get observation
    obs = test_reset()

    # Analyze resources
    analyze_resources(obs)

    # Test Task 1: DELETE unattached non-production resource
    task1_target = None
    for res in obs['resources']:
        if not res['is_production'] and res['last_accessed_days'] > 30:
            task1_target = res
            break

    if task1_target:
        action = {
            'resource_id': task1_target['id'],
            'action_type': 'delete'
        }
        test_step(action, f"DELETE {task1_target['id']} (Task 1 - Unattached)")
    else:
        print("No Task 1 target available\n")

    # Test Task 2: RIGHT_SIZE over-provisioned instance (need to find suitable one)
    # Let's reset a few times to get a better chance
    print("=== Trying to find Task 2 target ===")
    task2_target = None
    for attempt in range(3):
        r = requests.post('http://localhost:8000/reset')
        if r.status_code == 200:
            obs = r.json()
            task2_target = None
            for res in obs['resources']:
                if res['type'] in ['EC2', 'RDS'] and res['cpu_usage_avg'] < 5.0 and res['last_accessed_days'] > 7:
                    task2_target = res
                    break
            if task2_target:
                break

    if task2_target:
        action = {
            'resource_id': task2_target['id'],
            'action_type': 'right_size'
        }
        test_step(action, f"RIGHT_SIZE {task2_target['id']} (Task 2 - Over-provisioned)")
    else:
        print("No Task 2 target found after 3 attempts\n")

    # Test Task 3: TAG_FOR_REVIEW production idle resource
    task3_target = None
    for res in obs['resources']:
        if res['is_production'] and res['cpu_usage_avg'] < 5.0:
            task3_target = res
            break

    if task3_target:
        action = {
            'resource_id': task3_target['id'],
            'action_type': 'tag_for_review'
        }
        test_step(action, f"TAG_FOR_REVIEW {task3_target['id']} (Task 3 - Production idle)")
    else:
        print("No Task 3 target available\n")

    # Test dangerous action: DELETE production resource
    print("=== Testing Dangerous Action Penalty ===")
    prod_target = None
    for res in obs['resources']:
        if res['is_production']:
            prod_target = res
            break

    if prod_target:
        action = {
            'resource_id': prod_target['id'],
            'action_type': 'delete'
        }
        result = test_step(action, f"DELETE {prod_target['id']} (Production - Should be penalized)")
        # Check if dangerous penalty was applied
        if result['info'].get('dangerous'):
            print("Dangerous action correctly flagged")
        comp = result['info'].get('reward_components', {})
        if comp.get('dangerous_delete') == -0.5:
            print("Dangerous penalty of -0.5 applied")
    else:
        print("No production resource found for danger test\n")

    print("=" * 50)
    print("All tests completed!")

if __name__ == '__main__':
    main()