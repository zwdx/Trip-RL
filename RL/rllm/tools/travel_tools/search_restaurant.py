import pandas as pd  
from rllm.tools.tool_base import Tool   
import os 
import random
import math


RESTAURANT_PATH = os.getenv("RESTAURANT_PATH", "rllm/tools/travel_tools/database/clean_restaurant_2022.csv")
_RESTAURANT_DATA = None


def _load_restaurant_data():
    """Load and cache restaurant data once per process."""
    global _RESTAURANT_DATA
    if _RESTAURANT_DATA is None:
        _RESTAURANT_DATA = pd.read_csv(RESTAURANT_PATH).dropna()[['Name','Average Cost','Cuisines','Aggregate Rating','City']]
    return _RESTAURANT_DATA


class SearchRestaurant(Tool):
    NAME = "SearchRestaurant" 
    DESCRIPTION = "Find restaurant options in a specific city."

    def __init__(self, name: str = NAME, description: str = DESCRIPTION): 
        super().__init__(name=name, description=description) 
        self._data = _load_restaurant_data()

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
                            "description": "The city name to search for restaurants."
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
                return f"There is no restaurant in {city}. Please rethink your restaurant search and try again."
            
            results = [] 
            for i, row in enumerate(filtered_data.itertuples(index=False)):  
                # Original (compat) output format:
                cur_result = f"{i+1}. [{row[0]}] Avg. Cost ${row[1]}. Cuisines: {row[2]}. Rating: {row[3]}\n" 
                results.append(cur_result) 

            return f"A SearchRestaurant for city {city} found the following {len(results)} restaurants:\n" + "\n".join(results)
       
        except Exception as e:
            return f"Tool execution for SearchRestaurant failed with error: {e}"

    
_load_restaurant_data()


if __name__ == "__main__": 
    search_restaurant = SearchRestaurant()  

    for city in ["New York", "Los Angeles", "Vancouver"]:
        result = search_restaurant(city)
        print(result)
        print("-" * 50)
