from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    role: str = ""
    content: str = ""

    # "role": "user" or "developer" (developer for system instructions)
    def __init__(self, content: str = "", role: str = "user") -> None:
        super().__init__()
        self.role = role
        self.content = content

    def add_text(self, text: str) -> "Message":
        self.content += text
        return self

    def add_paragraph(self, header: str, content: str) -> "Message":
        self.content += f"{header}:\n" + content + "\n\n"
        return self

    def add_context(self, context: List["Message"]) -> "Message":
        history = ""
        for message in context:
            history += f"ROLE: " + message.role + "\nCONTENT:\n" + message.content + "\n"
        self.content += "CONTEXT:\n" + history + "\n"
        return self
