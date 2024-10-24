import pandas as pd

from datetime import time


def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Function to calculate the distance matrix
    # Get unique toll IDs
    ids=sorted(set(df['id_start']).union(set(df['id_end'])))
    
    # Initialize an empty DataFrame for the distance matrix
    distance_matrix=pd.DataFrame(index=ids, columns=ids, data=float('inf'))
    
    # Fill the diagonal with zeros
    for id in ids:
        distance_matrix.at[id, id]=0
    
    # Fill in known distances from the dataset
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end]=distance
        distance_matrix.at[id_end, id_start]=distance  # Ensure bidirectional distance
    
    # Floyd-Warshall Algorithm to find shortest cumulative distances
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.at[i, j]>distance_matrix.at[i, k]+distance_matrix.at[k, j]:
                    distance_matrix.at[i, j]=distance_matrix.at[i, k]+distance_matrix.at[k, j]
    
    return distance_matrix

# Calculate the distance matrix using the provided data
distance_matrix=calculate_distance_matrix(df)
distance_matrix

# Resulting matrix:
"""          1001400  1001402  1001404  1001406  1001408  1001410  1001412
1001400      0.0      9.7     29.9     45.9     67.6     78.7     94.3
1001402      9.7      0.0     20.2     36.2     57.9     69.0     84.6
1001404     29.9     20.2      0.0     16.0     37.7     48.8     64.4
1001406     45.9     36.2     16.0      0.0     21.7     32.8     48.4
1001408     67.6     57.9     37.7     21.7      0.0     11.1     26.7
1001410     78.7     69.0     48.8     32.8     11.1      0.0     15.6
1001412     94.3     84.6     64.4     48.4     26.7     15.6      0.0
"""



def unroll_distance_matrix(distance_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Function to unroll the distance matrix into id_start, id_end, and distance columns
    unrolled_data=[]

    # Iterate through the matrix and extract id_start, id_end, and distance
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start!=id_end:
                distance=distance_matrix.at[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])
    
    # Create a DataFrame with id_start, id_end, and distance
    unrolled_df=pd.DataFrame(unrolled_data, columns=["id_start", "id_end", "distance"])
    
    return unrolled_df

# Unroll the previously calculated distance matrix
unrolled_df=unroll_distance_matrix(distance_matrix)
unrolled_df.head()
# Resulting matrix:
"""   id_start   id_end  distance
  0   1001400  1001402       9.7
  1   1001400  1001404      29.9
  2   1001400  1001406      45.9
  3   1001400  1001408      67.6
  4   1001400  1001410      78.7
"""




def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Function to find ids within a 10% distance threshold of a reference id's average distance
    # Filter rows where the reference id is the starting point
    reference_df=unrolled_df[unrolled_df['id_start']==reference_id]
    
    # Calculate the average distance for the reference id
    avg_distance=reference_df['distance'].mean()

    # Define the 10% threshold (ceiling and floor)
    lower_bound=avg_distance*0.9
    upper_bound=avg_distance*1.1

    # Find ids within the threshold
    ids_within_threshold=unrolled_df.groupby('id_start')['distance'].mean()
    ids_within_threshold=ids_within_threshold[
        (ids_within_threshold>=lower_bound) & (ids_within_threshold<=upper_bound)
    ].index.tolist()

    # Sort the list of ids
    return sorted(ids_within_threshold)

# Test the function with a reference id from the dataset
reference_id=1001400
ids_within_threshold=find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
ids_within_threshold

# Resulting Matrix:
"""   id_start   id_end  distance
0   1001400  1001402       9.7
1   1001400  1001404      29.9
2   1001400  1001406      45.9
3   1001400  1001408      67.6
4   1001400  1001410      78.7
...."""




def calculate_toll_rate(unrolled_df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Function to calculate toll rates based on vehicle types
    # Define the rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type by multiplying distance by the corresponding rate
    for vehicle, rate in rates.items():
        unrolled_df[vehicle]=unrolled_df['distance']*rate
    
    return unrolled_df

# Apply the toll rate calculation on the unrolled dataframe
toll_rate_df=calculate_toll_rate(unrolled_df)
toll_rate_df.head()



def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    return df
