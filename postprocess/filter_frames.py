

def keep_frame(result, filter_rule='contains_child'):
    # result: list of body types (smpl or smil)
    if filter_rule == 'contains_both':
        return exists_child(result) and exists_adult(result)
    
    elif filter_rule == 'contains_child':
        return exists_child(result)
    
    elif filter_rule == 'contains_adult':
        return exists_adult(result)
    
    elif filter_rule == 'all':
        return True
    
    else:
        raise ValueError('filter_rule {} not recognized'.format(filter_rule))


def exists_adult(results):
    return 'smpl' in results

def exists_child(results):
    return 'smil' in results
