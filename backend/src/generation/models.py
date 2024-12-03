from pydantic import BaseModel # Most widely used data validation library for python


class Images(BaseModel):
    image_pixels: str
    shape: tuple
    
class InsertUserData(BaseModel):
    image_pixels:str
    shape: tuple
    caption: str