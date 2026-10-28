import os 
from rllm.tools.tool_base import Tool
import pandas as pd  
import random
import math


FLIGHT_PATH = os.getenv("FLIGHT_PATH", "rllm/tools/travel_tools/database/clean_Flights_2022.csv")
_FLIGHT_DATA = None


def _load_flight_data():
    """Load and cache flight data once per process so all tool instances share it."""
    global _FLIGHT_DATA
    if _FLIGHT_DATA is None:
        _FLIGHT_DATA = pd.read_csv(FLIGHT_PATH).dropna()[['Flight Number', 'Price', 'DepTime', 'ArrTime', 'ActualElapsedTime','FlightDate','OriginCityName','DestCityName','Distance']]
    return _FLIGHT_DATA


class SearchFlight(Tool):
    NAME = "SearchFlight"
    DESCRIPTION = "Search for flight information between cities on specific dates." 

    def __init__(self, name: str = NAME, description: str = DESCRIPTION):
        super().__init__(name=name, description=description) 
        self._data = _load_flight_data()

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
                        "departure": {
                            "type": "string", 
                            "description": "The departure city name."
                        },
                        "destination": {
                            "type": "string", 
                            "description": "The destination city name."
                        },
                        "date": {
                            "type": "string", 
                            "description": "Travel date in YYYY-MM-DD format."
                        },      
                    },
                    "required": ["departure", "destination", "date"],
                },
            }
        }
    
    def forward(self, departure: str, destination: str, date: str) -> str:
        try:
            # 1 - Data filtering 
            filtered_data = self.data[self.data["OriginCityName"] == departure]
            filtered_data = filtered_data[filtered_data["DestCityName"] == destination]
            filtered_data = filtered_data[filtered_data["FlightDate"] == date]

            # 2 - Format results 
            if len(filtered_data) == 0:
                return f"There is no available flight from {departure} to {destination} on {date}. Please rethink your flight search and try again."
        
            results = [] 
            for i, row in enumerate(filtered_data.itertuples(index=False)): 
                # Flight Number, Price, DepTime, ArrTime, ActualElapsedTime, FlightDate, OriginCityName, DestCityName, Distance
                cur_result = f"{i+1}. [{row[0]}] ${row[1]}. Departure: {row[2]}, Arrival: {row[3]}, Actual Elapsed Time: {row[4]}, Distance: {row[8]} miles\n" 
                results.append(cur_result) 
        
            return f"A SearchFlight for {departure} to {destination} on {date} found the following {len(filtered_data)} flights:\n" + "\n".join(results)
        except Exception as e:
            return f"Tool execution for SearchFlight failed with error: {e}"


_load_flight_data()  # eager load when module is imported to avoid spikes later


if __name__ == "__main__":
    search_flight = SearchFlight() 

    test_params = [
        {"departure": "New York", "destination": "London", "date": "2022-10-01"},
        {"departure": "Durango", "destination": "Denver", "date": "2022-04-04"},
        {"departure": "Washington", "destination": "Richmond", "date": "2022-05-05"},
        {"departure": "Buffalo", "destination": "Atlanta", "date": "2022-03-02"},
    ]

    for params in test_params:
        result = search_flight(**params)
        print(result)
        print("-" * 50)
