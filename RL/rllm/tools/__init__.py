from rllm.tools.code_tools import (
    PythonInterpreter,
)
from rllm.tools.registry import ToolRegistry
from rllm.tools.web_tools import (
    FirecrawlTool,
    GoogleSearchTool,
    TavilyExtractTool,
    TavilySearchTool,
)
from rllm.tools.travel_tools import (
    SearchCity,
    SearchAttraction,
    SearchRestaurant,
    SearchAccommodation,
    SearchFlight,
    GoogleDistanceMatrix,
)

# Define default tools dict
DEFAULT_TOOLS = {
    "python": PythonInterpreter,
    "google_search": GoogleSearchTool,
    "firecrawl": FirecrawlTool,
    "tavily-extract": TavilyExtractTool,
    "tavily-search": TavilySearchTool,

    # Travel tools
    "SearchCity": SearchCity,
    "SearchAttraction": SearchAttraction,
    "SearchRestaurant": SearchRestaurant,
    "SearchAccommodation": SearchAccommodation,
    "SearchFlight": SearchFlight,
    "GoogleDistanceMatrix": GoogleDistanceMatrix,
}

# Create the singleton registry instance and register all default tools
tool_registry = ToolRegistry()
tool_registry.register_all(DEFAULT_TOOLS)

__all__ = [
    "PythonInterpreter", "LocalRetrievalTool", "GoogleSearchTool", "FirecrawlTool", "TavilyExtractTool", "TavilySearchTool", "ToolRegistry", "tool_registry"
    
    # Travel Tools 
    "SearchCity", "SearchAttraction", "SearchRestaurant", "SearchAccommodation", "SearchFlight", "GoogleDistanceMatrix",
]
