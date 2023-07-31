import os


def get_env(key: str, default_value: str):
    """_summary_
    Fallback to default of the env get by key is not set or is empty string
    """
    env = os.environ.get(key)
    if env is None or len(env) == 0:
        return default_value
    else:
        return env
