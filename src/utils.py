import numpy as np
from collections import Counter 

def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]

def column2list(dataframe, column):
	"""Transform a dataframe column to a list.
	
	Args:
		dataframe: Name of a dataframe
		column (str): Name of the selected column.
	
	Return:
		Column as a list.
		
	"""
	return list(dataframe[column])

def unique_elements(values):
    """List the unique elements of a list."""
    return list(set(values))

def int2str(val):
    """Transform an integer to string type."""
    return str(val)

def count_co_occurrence(matches):
    """Find co-occurrence of elements in a dictionary.
    
    Args:
        matches (dict)
    
    Return:
        count_dict (dict): Dictionary where the values are Counter objects.
    
    """
    count_dict = {}
    for key, val in matches.items():
        count_dict[key] = Counter(val)
    
    return count_dict

def column_merge(df, broad_topics):
    """Merge the columns of a dataframe by summing up their values.

    Args:
        df (Pandas dataframe): Dataframe to transform.
        broad_topics (dict): Keys are the new column names and values are lists of columns names that will be transformed.

    Return:
        Dataframe with merged columns.

    """
    for broad_topic, topics in broad_topics.items():
        df[broad_topic] = np.sum([df[topic] for topic in topics], axis=0)
    
    return df
