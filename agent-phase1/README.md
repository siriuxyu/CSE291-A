# Agent Phase 1 - Archive

This directory contains an archived snapshot of the agent implementation from Phase 1 of the project.

## Purpose

This archive is maintained for:
- Comparison with future agent implementations
- Benchmarking and performance analysis
- Reference for understanding the evolution of the agent

## Important Notes

- **This is an archive copy**: Do not use this as the primary agent interface
- **For active development**: Use the main `agent/` directory and `run_agent.py` in the project root
- **Dependencies**: This archived agent still depends on the `tools/` directory in the project root

## Contents

- `graph.py` - Main agent graph implementation with ReAct pattern
- `context.py` - Context management for the agent
- `state.py` - State definitions for the conversation
- `utils.py` - Utility functions including logging and model loading
- `prompts.py` - System prompts
- `run_agent.py` - Interactive CLI interface (archived version)

## Running the Archived Agent

To run this archived version (for comparison purposes):

```bash
# From the project root
python agent-phase1/run_agent.py

# With debug mode
python agent-phase1/run_agent.py --debug
```

**Note:** This archived version uses its own isolated agent modules but shares the `tools/` directory from the project root. 

## Implementation Details

### Key Features (Phase 1)
- ReAct pattern
- Tool calling support
- User session isolation via thread IDs
- Structured logging
- Configurable model support

### Tools Available
- Calculator
- Weather information
- Web reader
- Translator
- File system search
