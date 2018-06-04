"""
T3S features extraction file.

The general steps are common to all models, but the specific implementations
of each depends on your model.
"""

def check_data(data):
    """
    Checks for the input data validity.

    Args:
        data: String to check.

    Returns:
        True if the data is valid for the model, False otherwise.
    """
    return '@' in data

def compute_features(data):
    """
    Computes the TensorFlow model features from the data.

    Args:
        data: String to compute the features from.

    Returns:
        A dictionary with the model's keys as keys and the computed features as
        values. The values must be lists, even with only one element (to match
        TensorFlow conditions).
    """
    # compute features values
    lp, domain = data.split('@')
    lp_length = len(lp)
    lp_num = sum(c.isdigit() for c in lp)
    lp_alpha = sum(c.isalpha() for c in lp)
    lp_other = lp_length - lp_num - lp_alpha

    # create features dictionary
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

def extract(data, debug=False):
    """
    Computes the TensorFlow model features from the data. This is specific to
    your model.

    Args:
        data: String to compute the features from.
        debug: A boolean flag to display or not the computed features.

    Returns:
        If the input data is not in the right form, returns None.
        Else, returns a dictionary with the model's keys as keys and the computed
        features as values.
    """
    # check for input validity
    if not check_data(data):
        return None

    # compute features
    features = compute_features(data)

    if debug:
        print('-' * 23)
        print('T3S computation result:')
        print('-' * 23)
        print('Input:', data)
        print('Result:', features, '\n')

    # return result
    return features
