---
type: "agent_requested"
description: "When developing agents"
---
- Create type-safe factory patterns for swappable embedding models and tokenizers
- Create clear abstractions for LLM providers, embedding models, and vector stores with well-defined interfaces
- Apply hexagonal architecture patterns to isolate core AI logic from external integrations
- Implement dependency injection patterns to improve testability and support multiple implementation strategies
- Design modular prompt templates with inheritance hierarchies and composition patterns
- Apply the principle of least knowledge (Law of Demeter) to reduce coupling between AI components
- Design clear upgrade paths for migration between model versions and embedding spaces
- Create fallback chains for graceful degradation when primary models or services fail
- Implement retry mechanisms with exponential backoff for transient LLM provider errors
- Follow the principle of "fail fast" for invalid inputs with comprehensive schema validation
- Implement proper handling of partial failures in batch operations
- Design timeouts at appropriate levels (request, operation, system) to prevent resource exhaustion
- Implement graceful handling of API quota limits and rate limiting responses
- Provide detailed error logging with contextual information while protecting sensitive data
