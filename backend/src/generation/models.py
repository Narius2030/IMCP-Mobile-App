from typing import Optional, List # Supports for type hints
from pydantic import BaseModel # Most widely used data validation library for python
from datetime import date, datetime


class Image(BaseModel):
    image_pixels: str
    shape: tuple