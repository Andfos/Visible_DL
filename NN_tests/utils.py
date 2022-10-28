import pandas as pd
import numpy as np

def create_connections_matrix(filepath):
    """ Read the design matrix from file in as a numpy ndarray """
    
    # Read in the excel file with the design matrices
    xlsx = pd.ExcelFile(filepath)
    
    # Read each sheet in as a 2D numpy array
    connections = []
    for sheet_name in xlsx.sheet_names:
        matrix = pd.read_excel("design_matrix.xlsx", 
                               sheet_name = sheet_name,  
                               index_col = False, 
                               dtype = float, 
                               skiprows = 0)
        matrix = matrix.to_numpy()
        connections.append(matrix)

    return connections
