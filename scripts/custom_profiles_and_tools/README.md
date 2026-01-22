# Custom Profiles & Tools

Create and run custom profiles and tools **outside** the library.


## Environment Variables

| Variable | Description |
|----------|-------------|
| `REACHY_MINI_CUSTOM_PROFILE` | Profile folder name (e.g., `custom_profile`) |
| `PROFILES_DIRECTORY` | Path to directory containing the profile folder |
| `TOOLS_DIRECTORY` | *(optional)* Path to directory containing custom tools |

## Create a Custom Profile

Create a folder with these files:

```
my_profile/
├── instructions.txt   # System prompt for the AI
├── tools.txt          # List of tools (one per line)
└── voice.txt          # Voice name (optional)
```

**`tools.txt`** - List built-in tools + your custom tools:
```
dance
play_emotion
my_custom_tool
```

## Create a Custom Tool

Create a Python file in your tools directory:

```python
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

class MyCustomTool(Tool):
    name = "my_custom_tool"
    description = "What this tool does"
    parameters_schema = {
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "..."},
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs):
        # Your logic here
        return {"status": "success"}
```

## Run

```bash
export REACHY_MINI_CUSTOM_PROFILE=my_profile
export PROFILES_DIRECTORY=/path/to/parent/folder
export TOOLS_DIRECTORY=/path/to/custom_tools
reachy-mini-conversation-app
```

check `run.sh` script for more details.

> ⚠️ **Warning**: If you have a `.env` file, it will overwrite these environment variables. Make sure your `.env` doesn't define `REACHY_MINI_CUSTOM_PROFILE`, `PROFILES_DIRECTORY`, or `TOOLS_DIRECTORY` unless intended.
