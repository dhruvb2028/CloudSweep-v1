from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class ResourceType(str, Enum):
    EC2 = "EC2"
    S3 = "S3"
    RDS = "RDS"

class ActionType(str, Enum):
    DELETE = "delete"
    RIGHT_SIZE = "right_size"
    TAG_FOR_REVIEW = "tag_for_review"
    NOTIFY_OWNER = "notify_owner"
    NO_OP = "no_op"

class CloudResource(BaseModel):
    id: str
    type: ResourceType
    cost_per_day: float
    cpu_usage_avg: float
    is_production: bool
    last_accessed_days: int

class Observation(BaseModel):
    resources: List[CloudResource]
    total_daily_cost: float
    day: int = Field(default=0, description="Current day in simulation")

class Action(BaseModel):
    resource_id: str
    action_type: ActionType
    parameters: dict = Field(default_factory=dict)

class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0, description="Reward value between 0 and 1")
    components: dict = Field(default_factory=dict, description="Breakdown of reward components")