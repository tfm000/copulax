from copulax import get_random_key


def test_get_random_key():
    """Tests the get_random_key function."""
    # testing api method
    key = get_random_key(max_attempts=10)

    # testing non-api method
    key = get_random_key(max_attempts=0)