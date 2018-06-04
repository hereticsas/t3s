"""
T3S features extraction file.
"""

def extract(data):
    """
    Computes the TensorFlow model features from the data. This is specific to
    your model.

    Args:
        data: String to compute the features from.

    Returns:
        If the input data is not in the right form, should return None.
        Else, returns a dictionary with the model's keys as keys and the computed
        features as values.
    """
    # check for input validity
    if '@' not in data:
        return None

    # compute features
    lp, domain = data.split('@')
    lp_length = len(lp)
    lp_num = sum(c.isdigit() for c in lp)
    lp_alpha = sum(c.isalpha() for c in lp)
    lp_other = lp_length - lp_num - lp_alpha

    # create features dictionary
    # (keys are the TensorFlow model keys, values must be lists, even with only 1 element)
    features = {
        'lp_length': [lp_length],
        'lp_alpha': [lp_alpha],
        'lp_num': [lp_num],
        'lp_other': [1],
        'domain_length': [len(domain)],
        'domain': [domain],
    }
    # return result
    return features
