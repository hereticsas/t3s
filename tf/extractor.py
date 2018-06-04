"""
T3S features extraction file.

The general steps are common to all models, but the specific implementations
of each depends on your model.
"""

def check_data(input):
    """
    Checks for the input validity.

    Args:
        input: String to check.

    Returns:
        True if the input is valid for the model, False otherwise.
    """
    return '@' in input

def compute_features(input):
    """
    Computes the TensorFlow model features from the input.

    Args:
        input: String to compute the features from.

    Returns:
        A dictionary with the model's keys as keys and the computed features as
        values.
    """
    # compute features values
    lp, domain = input.split('@')
    lp_length = len(lp)
    lp_num = sum(c.isdigit() for c in lp)
    lp_alpha = sum(c.isalpha() for c in lp)
    lp_other = lp_length - lp_num - lp_alpha

    # create features dictionary
    features = {
        'lp_length': lp_length,
        'lp_alpha': lp_alpha,
        'lp_num': lp_num,
        'lp_other': lp_other,
        'domain_length': len(domain),
        'domain': domain,
    }
    # return result
    return features

def extract(data, debug=False):
    """
    Computes the TensorFlow model features from the data. This is specific to
    your model.

    Args:
        data: String to compute the features from (can contain multiple examples,
            each separated by a ';' character).
        debug: A boolean flag to display or not the computed features.

    Returns:
        If the input data is not in the right form, returns None.
        Else, returns a dictionary with the model's keys as keys and the computed
        features as values.
    """
    # Split examples
    inputs = data.split(';')

    if debug:
        print('=' * 23)
        print('T3S computation result:')
        print('=' * 23)

    # Prepare results array
    examples_features = []
    # Compute each example
    for input in inputs:
        result = None

        # Check for input validity
        if check_data(input):
            # Compute features
            features = compute_features(input)

            if debug:
                print('Example #%3d:' % len(examples_features))
                print('-' * 13)
                print('Input:', input)
                print('Result:', features, '\n')

        examples_features.append(features)

    # return result
    return examples_features
