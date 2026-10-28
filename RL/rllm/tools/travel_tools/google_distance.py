import pandas as pd 
import numpy as np 
from rllm.tools.tool_base import Tool
import os 


DISTNACE_PATH = os.getenv("DISTNACE_PATH", "rllm/tools/travel_tools/database/distance.csv")
_DISTANCE_DATA = None


def check_value(val):
    return val is not None and val != "" and not val is np.nan


def _load_distance_data():
    """Load and cache distance data once per process."""
    global _DISTANCE_DATA
    if _DISTANCE_DATA is None:
        _DISTANCE_DATA = pd.read_csv(DISTNACE_PATH)
    return _DISTANCE_DATA


class GoogleDistanceMatrix(Tool):
    NAME = "GoogleDistanceMatrix"
    DESCRIPTION = "Calculate distance, travel time, and cost between two cities." 

    def __init__(self, name: str = NAME, description: str = DESCRIPTION): 
        super().__init__(name=name, description=description) 
        self._data = _load_distance_data()

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
                        "mode": {
                            "type": "string", 
                            "description": "Transportation mode: 'driving' or 'taxi'."
                        },
                    },
                    "required": ["departure", "destination", "mode"],
                }
            }
        }
    
    def forward(self, departure: str, destination: str, mode: str = "driving") -> str:
        try:
            # 1 - Data filtering 
            filtered_data = self.data[(self.data["origin"] == departure) & (self.data["destination"] == destination)] 
            
            # 2 - Format results
            if len(filtered_data):
                if not check_value(filtered_data['duration'].values[0]) or not check_value(filtered_data['distance'].values[0]):
                    result = f"Sorry, we cannot find the distance between {departure} and {destination}." 
                else:
                    duration = filtered_data['duration'].values[0] 
                    distance = filtered_data['distance'].values[0]
                    if "driving" in mode: 
                        cost = int(eval(distance.replace("km","").replace(",","").strip()) * 0.05)
                    else: # taxi
                        cost = int(eval(distance.replace("km","").replace(",","").strip()))
                    
                    if "day" in duration:
                        return "The required time exceeds a single day, we cannot provide the cost. Please consider an alternative transportation instead."

                    result = f"A GoogleDistanceMatrix found the following information:\n{mode}, from {departure} to {destination}, duration: {duration}, distance: {distance}, cost: ${cost}"
            else:
                result = "No valid information found. Please rethink your distance search and try again." 

            return result
            
        except Exception as e:
            return f"Tool execution for GoogleDistanceMatrix failed with error: {e}"
    
    def run_for_evaluation(self, origin: str, destination: str, mode: str = "driving") -> dict: 
        info = {
            "origin": origin,
            "destination": destination,
            "cost": None, 
            "duration": None,  
            "distance": None, 
        }
        response = self.data[(self.data["origin"] == origin) & (self.data["destination"] == destination)]
        if len(response): 
            if not check_value(response['duration'].values[0]) or not check_value(response['distance'].values[0]):
                return info 
            
            info["duration"] = response['duration'].values[0]
            info["distance"] = response['distance'].values[0]
            
            if 'day' not in info["duration"]: 
                if 'driving' in mode: 
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","").strip()) * 0.05)
                else: # taxi
                    info["cost"] = int(eval(info["distance"].replace("km","").replace(",","").strip()))

            return info 
        
        return info


_load_distance_data()


if __name__ == "__main__":
    google_distance = GoogleDistanceMatrix()  

    test_params = [
        {"departure": "Boston", "destination": "Washington", "mode": "driving"},
        {"departure": "Dallas", "destination": "San Antonio", "mode": "driving"},
        {"departure": "New York", "destination": "London", "mode": "driving"}, 
        {"departure": "Detroit", "destination": "Norfolk", "mode": "taxi"},
        {"departure": "Hartford", "destination": "Denver", "mode": "driving"},
    ]
    
    for params in test_params:
        result = google_distance(**params)
        print(result)
        print("-" * 50)
