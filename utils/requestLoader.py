import json
import numpy as np
import random

class RequestReader:
    def __init__(self, json_data):
        """
        Initializes the reader with JSON data received from a request body.
        :param json_data: JSON data as a string or dictionary representing the timeseries.
        """
        if isinstance(json_data, str):
            self.data = json.loads(json_data)  # Parse JSON from a string if necessary
        else:
            self.data = json_data  # Directly use the dictionary if already parsed
        self._process_data()  # Process data right after initialization
        self._current_index = 0
    
    def _process_data(self):
        """
        Processes the JSON timeseries data. This is a placeholder that can be extended in subclasses.
        """
        self.header = self.data["header"]
        self.data_array = np.array(self.data["data"])
        self.outcome = self.data.get("outcome", None)  # Optionally extract outcome if available

    def read_timeseries(self):
        """
        Returns the processed timeseries data.
        :return: Dictionary containing processed timeseries data.
        """
        return {
            "X": self.data_array,
            "header": self.header,
            "outcome": self.outcome  # Include additional relevant data fields
        }

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._data)

    def read_example(self, index):
        raise NotImplementedError()

    def read_next(self):
        to_read_index = self._current_index
        self._current_index += 1
        if self._current_index == self.get_number_of_examples():
            self._current_index = 0
        return self.read_example(to_read_index)
    
class InHospitalMortalityRequestReader(RequestReader):
    def _process_data(self):
        """
        Extends the base class method to include processing specific to in-hospital mortality data.
        """
        super()._process_data()  # Call the base class method to process common data
        self.patient_id = self.data.get("patient_id", "Unknown")  # Extract patient ID if available

    def read_timeseries(self):
        """
        Extends the base class output with specific in-hospital mortality data.
        """
        data = super().read_timeseries()  # Get the basic timeseries data from the superclass
        data["patient_id"] = self.patient_id
        return data
