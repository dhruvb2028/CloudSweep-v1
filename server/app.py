from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine import CloudSweepEngine
from models import Observation, Action, Reward

app = FastAPI(title="CloudSweep-v1 Environment", description="OpenEnv-compliant Cloud FinOps optimization environment")

# Global environment instance
env = CloudSweepEngine()

class StepRequest(BaseModel):
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict

@app.post("/reset", response_model=Observation)
async def reset_environment():
    """Reset the environment and return initial observation"""
    observation = env.reset()
    return observation

@app.post("/step", response_model=StepResponse)
async def step_environment(request: StepRequest):
    """Execute one step in the environment"""
    observation, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info
    )

@app.get("/observation", response_model=Observation)
async def get_observation():
    """Get current observation without stepping"""
    return env._get_observation()

@app.get("/state", response_model=Observation)
async def get_state():
    """Get current state of the environment"""
    return env.state()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CloudSweep-v1 Environment", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": "CloudSweep-v1"}

def main():
    """Main entry point for the application"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()