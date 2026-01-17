from pydantic import BaseModel

class Review(BaseModel):
    sentence: str