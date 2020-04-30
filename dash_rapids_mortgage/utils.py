from retrying import retry
scheduler_url = "localhost:8786"

# Names of float columns
float_columns = [
    'current_actual_upb', 'dti', 'borrower_credit_score', 'delinquency_12_prediction'
]


def compute_bounds(df, columns):
    """
    Compute the min/max bounds of select columns in a DataFrame
    Args:
        df: pandas or cudf DataFrame
        columns: list of columns to compute bounds on

    Returns:
        dict from input columns to (min, max) tuples
    """
    return {c: (df[c].min(), df[c].max()) for c in columns}


@retry(wait_exponential_multiplier=100, wait_exponential_max=2000, stop_max_delay=6000)
def get_dataset(client, name):
    return client.get_dataset(name)
