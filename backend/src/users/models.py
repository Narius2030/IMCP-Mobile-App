from typing import Optional, List # Supports for type hints
from pydantic import BaseModel # Most widely used data validation library for python
from enum import Enum # Supports for enumerations

# enum for gender
class Gender(str, Enum):
    male = "male"
    female = "female"

# enum for role
class Role(str, Enum):
    admin = "admin"
    image_team = "image-team"
    llm_team = "llm-team"

class User(BaseModel):
    username: str
    password: str
    gender: Gender
    email_address: Optional[str] = None
    phone_number:  Optional[str] = None
    roles: List[Role] # user can have several roles