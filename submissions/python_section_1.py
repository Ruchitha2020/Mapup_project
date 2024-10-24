from typing import Dict, List

import numpy as np

import pandas as pd

from itertools import permutations

from pprint import pprint

import re



def reverse_by_n_elements(lst, n):
    """
    Reverses the input list by groups of n elements.
    """
    result=[]
    i=0
    while i<len(lst):
        # Reverse the next group of n elements manually
        group=[]
        for j in range(min(n, len(lst)-i)):
            group.append(lst[i+j])
        
        # Manually reverse the group
        for j in range(len(group)-1, -1, -1):
            result.append(group[j])
        
        i+=n
    return result

# Test cases
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))      # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))               # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]




def group_by_length(lst):
    """
    Groups the strings by their length and returns a dictionary.
    """
    result={}
    
    # Iterate over each string in the list
    for s in lst:
        length=len(s)
        
        # Add the string to the appropriate length group in the dictionary
        if length in result:
            result[length].append(s)
        else:
            result[length]=[s]
    
    # Return the dictionary sorted by the keys (lengths)
    return dict(sorted(result.items()))

# Test cases
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3:['bat', 'car', 'dog'], 4:['bear'], 5:['apple'], 8:['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output: {3:['one', 'two'], 4:['four'], 5:['three']}




def flatten_dict(d, parent_key=''):
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    result = {}
    for k, v in d.items():
        # Create the new key by appending the parent key with a dot (if present)
        new_key=f"{parent_key}.{k}" if parent_key else k
        
        # If the value is a dictionary, flatten it recursively
        if isinstance(v, dict):
            result.update(flatten_dict(v, new_key))
        # If the value is a list, flatten each item and include the index
        elif isinstance(v, list):
            for i, item in enumerate(v):
                result.update(flatten_dict(item, f"{new_key}[{i}]"))
        # Otherwise, it's a base value, add it to the result
        else:
            result[new_key] = v
    
    return result

# Test case
data = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
print(flatten_dict(data))

# Output:
"""{
    "road.name": "Highway 1",
    "road.length": 350,
    "road.sections[0].id": 1,
    "road.sections[0].condition.pavement": "good",
    "road.sections[0].condition.traffic": "moderate"
}
"""




def unique_permutations(nums):
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    return list(map(list, set(permutations(nums))))

# Test case
output=unique_permutations([1, 1, 2])

# Print output in the desired format
pprint(output)

# Test case
[1, 1, 2]
# Output: 
"""[[1, 1, 2],
 [1, 2, 1],
 [2, 1, 1]]
"""



def find_all_dates(text):
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expression pattern to match the specified date formats
    pattern = r'(\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b)'
    return re.findall(pattern, text)

# Test case
text="I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output=find_all_dates(text)

# Print output
print(output)
# Output: ['23-08-1994', '08/23/1994', '1994.08.23']




    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
def haversine(coord1, coord2):
    # Haversine formula to calculate the distance between two points on the Earth
    R = 6371000                      # Radius of Earth in meters
    lat1, lon1=np.radians(coord1)
    lat2, lon2=np.radians(coord2)
    
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)* np.sin(dlon/2)**2
    c=2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def decode_polyline(polyline_str):
    # Decode polyline string into a list of (latitude, longitude) tuples
    coords=polyline.decode(polyline_str)
    
    # Create a DataFrame from the coordinates
    df=pd.DataFrame(coords, columns=["latitude", "longitude"])
    
    # Calculate distance from previous point
    df["distance"]=0.0                               # Initialize the distance column
    for i in range(1, len(df)):
        df.at[i, "distance"] = haversine(coords[i-1], coords[i])
    
    return df

# Example:
"""polyline_str = "u{~vFjqys@P?F?F?F?F?F?F?F?F?F?F?F?F?F?F?F?F?F"
   df = decode_polyline(polyline_str)

   print(df)
"""
# Output:
"""    latitude   longitude    distance
     0  38.5      -120.2        0.0
     1  40.5      -121.0        12345.67
     2  43.5      -123.0        23456.78
"""




def rotate_and_multiply_matrix(matrix):
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Step 1: Rotate the matrix by 90 degrees clockwise
    n=len(matrix)
    rotated_matrix=[list(row) for row in zip(*matrix[::-1])]
    
    # Step 2: Replace each element with the sum of all elements in the same row and column, excluding itself
    final_matrix=np.zeros((n, n), dtype=int)  # To store the result matrix

    for i in range(n):
        for j in range(n):
            # Calculate the sum of the row and column
            row_sum=sum(rotated_matrix[i])
            col_sum=sum(rotated_matrix[k][j] for k in range(n))
            # Exclude the current element from the sum
            final_matrix[i][j]=row_sum+col_sum-rotated_matrix[i][j]
    
    return final_matrix.tolist()

# Test case
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_transform(matrix)
print(result)
# Output:
# Input matrix:
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

# After rotating by 90 degrees clockwise:
rotated_matrix = [[7, 4, 1],
                  [8, 5, 2],
                  [9, 6, 3]]

# Final Output:
[[22, 19, 16],
 [23, 20, 17],
 [24, 21, 18]]





def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()
