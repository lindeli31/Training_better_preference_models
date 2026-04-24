from pydantic import BaseModel


class Response(BaseModel):
    message: str = ""
    reasoning: str | None = None