from get_env import get_env_or_none

def truncate(string, beginning=True):
    """Shorten the given string to the given length.

    :Parameters:
        length (int) = The maximum allowed length before truncating.
        beginning (bool) = Trim starting chars, else; ending.

    :Return:
        (str)

    ex. call: truncate('12345678', 4)
        returns: '5678'
    """
    TRUNCATE_PROMPT_LENGTH = get_env_or_none("TRUNCATE_PROMPT_LENGTH")
    if (TRUNCATE_PROMPT_LENGTH is None):
      return string
    length = int(TRUNCATE_PROMPT_LENGTH)
    if len(string) > length:
        # trim starting chars.
        if beginning:
            string = string[-length:]
        # trim ending chars.
        else:
            string = string[:length]
    return string
