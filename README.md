# LangchainResearchAgent 🕵️‍♂️

A comprehensive research-assisting agent built with the **LangChain** and **LangGraph** frameworks. This agent is designed to autonomously plan, research, and summarize information using a ReAct (Reason + Act) loop.


## 🧠 Architecture

The agent utilizes a cyclical **ReAct** architecture. Unlike simple linear chains, this agent can loop—reasoning, acting, and observing—until it determines it has sufficient information to answer the user's request.

```mermaid
graph TD
    User(User Input) --> NodeReasoning[<b>LLM Reasoning</b><br/>Plan & Decide]
    
    NodeReasoning -->|Decides to Act| Check{Check: <br/>Is tool needed?}
    
    Check -- Yes --> NodeTools[<b>Tool Execution</b><br/>Act & Observe]
    NodeTools -->|Observation Result| Memory[(<b>Memory</b><br/>Short-term State)]
    
    Memory --> NodeReasoning
    
    Check -- No (Done) --> FinalResponse(Final Answer)
    
    style NodeReasoning fill:#e1f5fe,stroke:#01579b
    style NodeTools fill:#fff9c4,stroke:#fbc02d
    style Memory fill:#f3e5f5,stroke:#7b1fa2