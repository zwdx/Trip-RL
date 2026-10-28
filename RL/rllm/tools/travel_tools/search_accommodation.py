import pandas as pd 
from rllm.tools.tool_base import Tool 
import os 


ACCOMMODATION_PATH = os.getenv("ACCOMMODATION_PATH", "rllm/tools/travel_tools/database/clean_accommodations_2022.csv")
_ACCOMMODATION_DATA = None


def _load_accommodation_data():
    """Load and cache accommodation data once per process."""
    global _ACCOMMODATION_DATA
    if _ACCOMMODATION_DATA is None:
        _ACCOMMODATION_DATA = pd.read_csv(ACCOMMODATION_PATH).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
    return _ACCOMMODATION_DATA


class SearchAccommodation(Tool):
    NAME = "SearchAccommodation"
    DESCRIPTION = "Find accommodation options in a specific city." 
    
    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description)
        self._data = _load_accommodation_data()

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
                            "description": "The city name to search for accommodations."
                        },
                    },
                    "required": ["city"],
                },
            }
        }
    
    def forward(self, city: str) -> str: 
        try:
            # 1 - Data filtering 
            filtered_data = self.data[self.data["city"] == city]

            # 2 - Format results
            if len(filtered_data) == 0:
                return f"There is no accommodation in {city}. Please rethink your accommodation search and try again."

            results = [] 
            for i, row in enumerate(filtered_data.itertuples(index=False)):  
                # Original (compat) output format:
                cur_result = f"{i+1}. [{row[0]}] ${row[1]}. Room Type: {row[2]}, House Rules: {row[3]}, Minimum Nights: {row[4]}, Maximum Occupancy: {row[5]}, Review Rate Number: {row[6]}\n"
                results.append(cur_result) 
            return f"A SearchAccommodation for {city} found the following {len(results)} accommodations:\n" + "\n".join(results)
       
        except Exception as e:
            return f"Tool execution for SearchAccommodation failed with error: {e}"


_load_accommodation_data()


if __name__ == "__main__":
    search_accommodation = SearchAccommodation()
    
    for city in ["New York", "Champaign", "Los Angeles", "Seattle", "Vancouver"]:
        result = search_accommodation(city)
        print(result)
        print("-" * 50)
