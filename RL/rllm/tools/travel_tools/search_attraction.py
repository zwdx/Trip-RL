import pandas as pd  
from rllm.tools.tool_base import Tool  
import os 


ATTRACTION_PATH = os.getenv("ATTRACTION_PATH", "rllm/tools/travel_tools/database/attractions.csv")
_ATTRACTION_DATA = None


def _load_attraction_data():
    """Load and cache attraction data once per process."""
    global _ATTRACTION_DATA
    if _ATTRACTION_DATA is None:
        _ATTRACTION_DATA = pd.read_csv(ATTRACTION_PATH).dropna()[['Name','Latitude','Longitude','Address','Phone','Website',"City"]]
    return _ATTRACTION_DATA


class SearchAttraction(Tool): 
    NAME = "SearchAttraction"
    DESCRIPTION = "Find tourist attractions in a specific city." 

    def __init__(self, name: str = NAME, description: str = DESCRIPTION): 
        super().__init__(name=name, description=description) 
        self._data = _load_attraction_data()

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
                        "city": {
                            "type": "string", 
                            "description": "The city name to search for attractions."
                        },
                    },
                    "required": ["city"],
                }
            }
        }

    def forward(self, city: str) -> str:
        try:
            # 1 - Data filtering 
            filtered_data = self.data[self.data["City"] == city] 

            # 2 - Format results 
            if len(filtered_data) == 0:
                return f"There is no attraction in {city}. Please rethink your attraction search and try again."
            
            results = []  
            for i, row in enumerate(filtered_data.itertuples(index=False)):  
                # Original (compat) output format:
                cur_result = f"{i+1}. [{row[0]}]({row[5]}) Address: {row[3]}. Contact: {row[4]}\n"
                results.append(cur_result) 
            return f"A SearchAttraction for city {city} found the following {len(results)} attractions:\n" + "\n".join(results)
       
        except Exception as e:
            return f"Tool execution for SearchAttraction failed with error: {e}"


_load_attraction_data()


if __name__ == "__main__":
    search_attraction = SearchAttraction() 

    for city in ["New York", "Champaign", "Los Angeles", "Chicago"]:
        result = search_attraction(city)
        print(result)
        print("-" * 50)
