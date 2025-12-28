import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage

# This defines the data structure passed between nodes
class AgentState(TypedDict):
    # 'operator.add' means: when a node returns a message, append it to the list
    messages: Annotated[List[BaseMessage], operator.add]