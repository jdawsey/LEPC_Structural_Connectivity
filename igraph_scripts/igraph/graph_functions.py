import pandas as pd
import igraph as ig
import numpy as np
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point, LineString
import geopandas as gpd



"""
This function should allow for creating a graph that generates for a
specified threshld distance.
given_df = a dataframe with coordinates
threshold_distance = the threshold distance desired for a graph to be generated
"""
def threshold_graph(given_df, threshold_distance):
    # List to hold all pairwise distances
    distances = []

    # Calculate all pairwise distances
    for i in range(len(given_df)):
        for j in range(i + 1, len(given_df)):
            easting_diff = given_df.iloc[i]['x_easting'] - given_df.iloc[j]['x_easting']
            northing_diff = given_df.iloc[i]['y_northing'] - given_df.iloc[j]['y_northing']
            distance = np.sqrt(easting_diff**2 + northing_diff**2)
            distances.append(distance)

    # Sort distances
    distances.sort()

    # Function to check if the graph is connected for a given threshold
    g_temp = ig.Graph()
    g_temp.add_vertices(len(given_df))
    g_temp.vs["name"] = given_df['lek_id'].tolist()

    for i in range(len(given_df)):
        for j in range(i + 1, len(given_df)):
            easting_diff = given_df.iloc[i]['x_easting'] - given_df.iloc[j]['x_easting']
            northing_diff = given_df.iloc[i]['y_northing'] - given_df.iloc[j]['y_northing']
            distance = np.sqrt(easting_diff**2 + northing_diff**2)

            if distance <= threshold_distance:
                g_temp.add_edge(i, j)

    return g_temp



"""
This function should allow for the automatic finding of a the coalescence distance
from a given dataframe.
"""
def coalescence_graph(given_df):
    # List to hold all pairwise distances
    distances = []

    # Calculate all pairwise distances
    for i in range(len(given_df)):
        for j in range(i + 1, len(given_df)):
            easting_diff = given_df.iloc[i]['x_easting'] - given_df.iloc[j]['x_easting']
            northing_diff = given_df.iloc[i]['y_northing'] - given_df.iloc[j]['y_northing']
            distance = np.sqrt(easting_diff**2 + northing_diff**2)
            distances.append(distance)

    # Sort distances
    distances.sort()

    # Function to check if the graph is connected for a given threshold
    def is_connected(threshold):
        g_temp = ig.Graph()
        g_temp.add_vertices(len(given_df))
        g_temp.vs["name"] = given_df['lek_id'].tolist()

        for i in range(len(given_df)):
            for j in range(i + 1, len(given_df)):
                easting_diff = given_df.iloc[i]['x_easting'] - given_df.iloc[j]['x_easting']
                northing_diff = given_df.iloc[i]['y_northing'] - given_df.iloc[j]['y_northing']
                distance = np.sqrt(easting_diff**2 + northing_diff**2)

                if distance <= threshold:
                    g_temp.add_edge(i, j)

        return g_temp.is_connected(), g_temp

    # Binary search for the minimum threshold distance
    low, high = 0, len(distances) - 1
    result = -1  # Default if no threshold found
    final_graph = None

    while low <= high:
        mid = (low + high) // 2
        threshold_distance = distances[mid]

        connected, temp_graph = is_connected(threshold_distance)
        if connected:
            result = threshold_distance  # Update result
            final_graph = temp_graph  # Store the final graph
            high = mid - 1  # Try for a smaller threshold
        else:
            low = mid + 1  # Increase threshold
    
    print(f'The coalescence distance is {result} meters')
    return final_graph



"""
This function allows for the exporting the edges from a given graph.
graph = the given graph
given_df = a dataframe for coordinates to come from
given_path = the desired path for export.
"""
def edge_shp_export(graph, given_df, given_path):
    # Prepare edges for exporting
    edges_data = []
    for edge in graph.get_edgelist():
        lek1_index, lek2_index = edge
        lek1 = given_df.iloc[lek1_index]
        lek2 = given_df.iloc[lek2_index]
        
        # Create a line geometry for the edge
        line_geom = LineString([(lek1['x_easting'], lek1['y_northing']),
                                (lek2['x_easting'], lek2['y_northing'])])
        
        # Calculate the length of the edge
        edge_length = line_geom.length  # This gives the length in the same units as your coordinates (meters)

        # Append the geometry and length to the edges data
        edges_data.append({'geometry': line_geom, 'lek1': lek1['lek_id'], 'lek2': lek2['lek_id'], 'length': edge_length})

    # Create a GeoDataFrame
    edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:32613')  # Set the appropriate CRS for UTM 13N

    # Export to shapefile
    edges_gdf.to_file(given_path, driver='ESRI Shapefile')
    print('edges exported to shapefile')