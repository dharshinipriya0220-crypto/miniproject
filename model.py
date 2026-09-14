from pydantic import BaseModel,Field

class ChatRequest(BaseModel):
    chat:str=Field(
         min_length=1,
    )

    system_prompt:str="you r a Ai assistant"

class ChatResponse(BaseModel):
     response:str