from rllm.tools.tool_base import Tool
import os 
from collections import defaultdict 


CITY_PATH = os.getenv("CITY_PATH", "rllm/tools/travel_tools/database/background/citySet_with_states.txt")
_CITY_DATA = None


def _load_city_data():
    """Load and cache city-state mappings once per process."""
    global _CITY_DATA
    if _CITY_DATA is None:
        city_state_data = open(CITY_PATH, "r").read().strip().split("\n") 
        cache = defaultdict(list)
        for unit in city_state_data: 
            city, state = unit.split("\t")  
            cache[state].append(city)
        _CITY_DATA = cache
    return _CITY_DATA


class SearchCity(Tool): 
    NAME = "SearchCity"
    DESCRIPTION = "Find cities within a specific state." 

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description)
        self._data = _load_city_data()

    @property
    def data(self):
        return self._data
    
    @property  
    def json(self):
        return {
            "type": "function", 
            "function": {
                "name": self.name,  
                "description": self.description,  
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "state": {
                            "type": "string", 
                            "description": "The state name to search for cities."
                        }
                    },
                    "required": ["state"],
                },
            }
        }
    
    def forward(self, state: str) -> str: 
        try:
            results = self.data.get(state, []) 

            if len(results) == 0:
                return f"There is no city in {state}. Please rethink your city search and try again." 
            
            return f"A SearchCity for state {state} found the following {len(results)} cities:\n" + ", ".join(results)
        except Exception as e: 
            return f"Tool execution for SearchCity failed with error: {e}"


_load_city_data()


if __name__ == "__main__":
    search_city = SearchCity() 

    test_params = ["Banana", "California", "Texas"]

    for param in test_params:
        result = search_city(param)
        print(result)
        print("-" * 50)
