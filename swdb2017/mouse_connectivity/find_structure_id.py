def find_structure_id(structure_tree,query):
    """
    Given a partial structure name as input, will print out all matching
    names together with their respective ids.
    
    A matching occurs for all names that have query as a substring (case
    insensitive)
    """
    for k,name in structure_tree.get_name_map().iteritems():
        if name.lower().find(query) >= 0:
            print k,'-',name