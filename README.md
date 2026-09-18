# Sezzle Support Agent

Small LLM-powered customer support assistant built for the Sezzle AI Engineer
take-home challenge.

The agent uses a local `qwen3.5:4b` model through Ollama, grounded policy
retrieval, authorized order lookup, structured LLM decisions, and deterministic
validation around the model.

## Architecture

```text
Question + user_id
        |
        +---- Policy retrieval
        |       |
        |   lexical + semantic
        |
        +---- Authorized order lookup
        |       |
        |   only this user's data
        |
        v
     qwen3.5:4b
        |
 structured decision
        |
        v
 route + grounded answer