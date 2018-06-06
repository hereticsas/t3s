from t3s import T3SExtractor

class CustomExtractor(T3SExtractor):

    def check_data(self, input):
        return '@' in input

    def compute_features(self, input):
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

        return features

    def error_formatting(self):
        return 'Please enter emails in the form: "username@domain".'
