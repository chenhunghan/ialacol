import os


def get_default_thread() -> int:
    """_summary_
    Automatically get the default number of threads to use for generation
    """
    count = os.cpu_count()
    if count is not None:
        return int(count / 2)
    else:
        return 8
