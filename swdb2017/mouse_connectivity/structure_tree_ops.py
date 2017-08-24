import numpy as np

def get_acronym_name_map(structure_tree):
    '''Builds a dictionary mapping structure acronyms to names.

    Parameters
    ----------
    structure_tree : StructureTree
        The complete structure tree.

    Returns
    -------
    acronym_name_map : dict
        Keys are acronyms (str); values are names (str).

    '''

    acronym_map = structure_tree.get_id_acronym_map()
    name_map = structure_tree.get_name_map()

    return {k:name_map[v] for k,v in acronym_map.iteritems()}

