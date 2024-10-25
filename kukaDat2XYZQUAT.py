#CHAT GPT!

import re
import math
import json
import numpy as np

class E6PosExtractor:
    def __init__(self):
        self.positions = {}

    def extract_positions(self, file_content):
        # Split file content by lines
        lines = file_content.splitlines()
        
        # Regular expression to match lines starting with "DECL E6POS" and capture the identifier and content
        pattern = r"^DECL E6POS\s+(\w+)={(.*)}"
        
        for line in lines:
            # Search for lines that match the pattern
            match = re.match(pattern, line)
            if match:
                # Capture the position identifier (e.g., XP1, XP2, XP3, etc.)
                position_id = match.group(1)
                # Capture the content within the curly braces
                content = match.group(2)
                # Parse the content into a dictionary
                parsed_data = self._parse_content(content)
                
                # Convert parsed data to position and quaternion format
                position_mm = [parsed_data['X'], parsed_data['Y'], parsed_data['Z']]
                # quaternion = self._euler_to_quaternion(parsed_data['A'], parsed_data['B'], parsed_data['C'])
                
                # # Store the position and quaternion data
                # self.positions[position_id] = {
                #     'position_mm': position_mm,
                #     'quaternion': quaternion
                # }
                rodrigues_vector = self._euler_to_rodrigues(parsed_data['A'], parsed_data['B'], parsed_data['C'])
                
                # Store the position (in mm) and Rodrigues rotation vector data
                self.positions[position_id] = {
                    'position_mm': position_mm,
                    'rodrigues': rodrigues_vector
                }

    def _parse_content(self, content):
        # Regular expression to match key-value pairs (e.g., X 387.504486)
        pair_pattern = r"(\w+)\s+([-+]?[0-9]*\.?[0-9]+)"
        parsed_data = {}
        
        for match in re.finditer(pair_pattern, content):
            # Extract key and value, convert value to float
            key, value = match.groups()
            parsed_data[key] = float(value)
        
        return parsed_data

    def _euler_to_quaternion(self, a, b, c):
        # Convert angles from degrees to radians
        a_rad = math.radians(a)
        b_rad = math.radians(b)
        c_rad = math.radians(c)
        
        # Compute individual quaternions for Rz * Ry * Rx rotation order
        cz = math.cos(a_rad / 2)
        sz = math.sin(a_rad / 2)
        cy = math.cos(b_rad / 2)
        sy = math.sin(b_rad / 2)
        cx = math.cos(c_rad / 2)
        sx = math.sin(c_rad / 2)
        
        # Quaternion multiplication for Rz * Ry * Rx
        qw = cz * cy * cx + sz * sy * sx
        qx = cz * cy * sx - sz * sy * cx
        qy = sz * cy * sx + cz * sy * cx
        qz = sz * cy * cx - cz * sy * sx
        
        return [qw, qx, qy, qz]

    def _euler_to_rodrigues(self, a, b, c):
        # Convert angles from degrees to radians
        a_rad = math.radians(a)
        b_rad = math.radians(b)
        c_rad = math.radians(c)
        
        # Calculate the rotation matrix for ZYX rotation order
        Rz = np.array([
            [math.cos(a_rad), -math.sin(a_rad), 0],
            [math.sin(a_rad),  math.cos(a_rad), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [math.cos(b_rad), 0, math.sin(b_rad)],
            [0, 1, 0],
            [-math.sin(b_rad), 0, math.cos(b_rad)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(c_rad), -math.sin(c_rad)],
            [0, math.sin(c_rad),  math.cos(c_rad)]
        ])
        
        # Combined rotation matrix for Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        
        # Compute the angle theta
        theta = math.acos((np.trace(R) - 1) / 2)
        
        # Compute the Rodrigues rotation vector components
        if theta != 0:
            rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
            ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
            rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
            rodrigues_vector = [theta * rx, theta * ry, theta * rz]
        else:
            # If theta is 0, there is no rotation, so the Rodrigues vector is [0, 0, 0]
            rodrigues_vector = [0, 0, 0]
        
        return rodrigues_vector
    

    def save_to_json(self, filename):
        # Serialize self.positions dictionary to a JSON file
        with open(filename, 'w') as json_file:
            json.dump(self.positions, json_file, indent=4)



class Pars:
    inputFile=r'.\handeye\1\handEyePts.dat'
    outputFile=r'.\handeye\1\poses.json'
    
if __name__=='__main__':

    # Initialize and use the extractor
    extractor = E6PosExtractor()
    f = open(Pars.inputFile,'r')
    if f is None:
        exit(1)
    file_content = f.read()
    extractor.extract_positions(file_content)

    # Save to JSON file
    extractor.save_to_json(Pars.outputFile)

    print(f"Data saved to: {Pars.outputFile}")
