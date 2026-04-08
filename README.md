# CloudSweep-v1: OpenEnv-Compliant Cloud FinOps Environment

A reinforcement learning environment for cloud cost optimization where agents identify and act on inefficient cloud resources to reduce expenses while avoiding dangerous actions.

## Overview

CloudSweep-v1 simulates a Cloud FinOps dashboard where an RL agent must optimize cloud costs by:
- Identifying and deleting unattached non-production resources
- Right-sizing over-provisioned compute instances  
- Handling production idle resources appropriately (tag/notify, don't delete)

The environment follows the OpenEnv specification and is ready for Hugging Face Spaces deployment.

## Features

- **OpenEnv Compliant**: Full implementation of the OpenEnv interface
- **Three Progressive Tasks**: Easy → Medium → Hard difficulty progression
- **Realistic Cloud Resources**: EC2 instances, S3 buckets, and RDS databases with realistic attributes
- **Proper Reward Function**: 
  - +0.1 reward per $10 saved (cost optimization)
  - -0.5 penalty for dangerous actions (deleting production resources)
  - Small rewards for correct identification before action
- **Baseline Agent**: Included inference script with OpenAI API integration
- **Deployment Ready**: Dockerfile for Hugging Face Spaces
- **Comprehensive Testing**: Built-in validation and test scripts

## Environment Details

### State Space (Observation)
- `resources`: List of cloud resource objects
- `total_daily_cost`: Sum of all resource daily costs
- `day`: Current simulation day (0-30)

Each resource has:
- `id`: Unique identifier (e.g., "EC2-001")
- `type`: Resource type (EC2, S3, or RDS)
- `cost_per_day`: Daily cost in USD
- `cpu_usage_avg`: Average CPU utilization percentage (0 for S3)
- `is_production`: Boolean indicating production vs. non-production
- `last_accessed_days`: Days since last access

### Action Space
- `resource_id`: Target resource identifier
- `action_type`: One of:
  - `delete`: Remove resource entirely
  - `right_size`: Reduce resource size/cost (EC2/RDS only)
  - `tag_for_review`: Mark for team review
  - `notify_owner`: Send notification to resource owner
  - `no_op`: Take no action

### Reward Function
- **Cost Savings**: +0.1 reward per $10 saved daily (max 0.5 per step)
- **Dangerous Actions**: -0.5 penalty for deleting production resources
- **Correct Identification**: 
  - +0.05 for tagging production idle resources
  - +0.03 for notifying owner of production idle resources
- **Invalid Actions**: Small negative rewards for inappropriate actions
- **Range**: Final reward clipped to [0.0, 1.0] as per OpenEnv spec

### Episodes
- Maximum 30 steps (days)
- Terminates early if all resources are processed
- Success criteria vary by task (see tasks below)

## Tasks

### Task 1: Clean up Unattached Resources (Easy)
**Objective**: Identify and delete resources that have not been accessed for >30 days and are not production.
**Success Criteria**:
- Delete at least 80% of eligible unattached non-production resources
- Do not delete any production resources
- Achieve minimum cost savings of $50

### Task 2: Right-size Over-provisioned Instances (Medium)
**Objective**: Identify EC2/RDS instances with CPU <5% for 7+ days and reduce their size/cost.
**Success Criteria**:
- Right-size at least 70% of eligible over-provisioned instances
- Do not right-size instances with CPU >=5% or accessed within 7 days
- Achieve minimum cost savings of $30 from right-sizing

### Task 3: Handle Production Idle Resources (Hard)
**Objective**: Identify production instances that are idle (CPU <5%) and take appropriate action (tag for review or notify owner) without deleting.
**Success Criteria**:
- Correctly tag/notify at least 80% of eligible production idle resources
- Do not delete any production resources
- Take appropriate action on all identified production idle resources

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git (optional, for cloning)

### Local Installation
1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd CloudSweep-v1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import fastapi, uvicorn, pydantic; print('Dependencies installed successfully')"
   ```

## Running the Environment

### Method 1: Direct Execution (Development/Testing)
```bash
# Start the FastAPI server
python main.py

# Server will be available at: http://localhost:8000
```

### Method 2: Using Docker
```bash
# Build the Docker image
docker build -t cloudsweep-v1 .

# Run the container
docker run -p 8000:8000 cloudsweep-v1

# Server will be available at: http://localhost:8000
```

### Method 3: Hugging Face Spaces
1. Push this repository to Hugging Face Spaces
2. The Dockerfile will automatically build and deploy the environment
3. Access via your Spaces URL

## Testing the Environment

### Health Check
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","environment":"CloudSweep-v1"}
```

### Reset Environment
```bash
curl -X POST http://localhost:8000/reset
# Returns initial observation with 10-20 cloud resources
```

### Execute an Action
```bash
# Example: Delete an unattached resource
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"resource_id":"EC2-005","action_type":"delete"}}'
```

### Run the Baseline Inference Script
```bash
# Simple run (uses rule-based policy)
python inference.py 2>&1

# With specific task description
TASK_DESCRIPTION="Clean up unattached resources" python inference.py 2>&1

# Expected output format:
# [START] episode_id:1234 task:Task description
# [STEP] step:1 action:delete:EC2-005 reward:0.450 total_reward:0.450
# [STEP] step:2 action:no_op:EC2-000 reward:0.000 total_reward:0.450
# ...
# [END] episode_id:1234 total_reward:0.850 length:25 success:true
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Reset environment and return initial observation |
| POST | `/step` | Execute an action and return (observation, reward, done, info) |
| GET | `/observation` | Get current observation without stepping |
| GET | `/state` | Get current environment state (alias for observation) |

## Project Structure

```
CloudSweep-v1/
├── models.py           # Pydantic data models
├── engine.py           # Simulation logic and environment core
├── main.py             # FastAPI server implementation
├── inference.py        # Baseline agent with OpenAI integration
├── openenv.yaml        # Environment metadata and task definitions
├── Dockerfile          # Containerization for Hugging Face Spaces
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── test_env.py         # Comprehensive test suite (development)
└── test_openenv.py     # OpenEnv compliance validation (development)
```

## Dockerfile Details

The provided Dockerfile creates a minimal container for deployment:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

To build and run:
```bash
docker build -t cloudsweep-v1 .
docker run -p 8000:8000 cloudsweep-v1
```

## OpenAI Integration (Baseline Agent)

The `inference.py` script includes a baseline agent that:
1. Attempts to use OpenAI API for action selection (when API key available)
2. Falls back to rule-based policy for demonstration
3. Emits required OpenEnv log formats: [START], [STEP], [END]

To use with actual OpenAI API:
1. Set environment variable: `OPENAI_API_KEY=your_key_here`
2. The script will automatically use the API for action selection

## Validation & Testing

Run the included test suite to verify functionality:
```bash
python test_env.py
```

This tests:
- Health endpoint
- Environment reset
- All three task types
- Dangerous action penalties
- Resource analysis and reporting

## Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face (select Docker template)
2. Push this repository to the Space's git repository
3. Hugging Face will automatically build the Docker image
4. The environment will be accessible at your Space's URL

## Troubleshooting

### Common Issues

**Port already in use**:
- Change port in main.py or use different port
- Kill existing processes: `taskkill /f /im python.exe` (Windows) or `pkill -f uvicorn` (Linux/Mac)

**Module not found errors**:
- Ensure you're using the correct Python environment
- Reinstall dependencies: `pip install -r requirements.txt`

**Docker build fails**:
- Check Docker daemon is running
- Ensure sufficient disk space and memory

**No matching resources for tasks**:
- The environment uses random resource generation
- Try resetting multiple times to get suitable resource distributions
- Tasks are designed to be solvable with reasonable probability

## License & Attribution

This environment was created for the OpenEnv Cloud FinOps challenge. Feel free to use and modify for research and educational purposes.

## Contact

For issues or questions regarding this environment, please refer to the project repository or open an issue in the appropriate forum.