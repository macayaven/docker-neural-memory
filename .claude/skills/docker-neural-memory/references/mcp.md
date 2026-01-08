# MCP Protocol Implementation

## Server Setup

```python
"""MCP server for neural memory."""

from mcp.server import Server
from mcp.types import Tool, TextContent

def create_server(memory, checkpoint_manager):
    server = Server("neural-memory")
    
    @server.list_tools()
    async def list_tools():
        return [
            Tool(name="observe", description="Learn from content", inputSchema={...}),
            Tool(name="infer", description="Generate from learned patterns", inputSchema={...}),
            Tool(name="surprise", description="Measure novelty", inputSchema={...}),
            Tool(name="consolidate", description="Compress patterns", inputSchema={...}),
            Tool(name="checkpoint", description="Save state", inputSchema={...}),
            Tool(name="restore", description="Load state", inputSchema={...}),
            Tool(name="fork", description="Branch state", inputSchema={...}),
        ]
    
    @server.call_tool()
    async def call_tool(name, arguments):
        if name == "observe":
            result = memory.observe(arguments["context"])
            return [TextContent(type="text", text=json.dumps(result))]
        # ... other tools
    
    return server
```

## HTTP Alternative

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/observe")
async def observe(request: ObserveRequest):
    return memory.observe(request.context)

@app.post("/infer")
async def infer(request: InferRequest):
    return memory.infer(request.query)

@app.get("/health")
async def health():
    return {"status": "healthy", "weight_hash": memory.get_weight_hash()}
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-v", "weights:/app/weights", "neural-memory:latest"]
    }
  }
}
```
