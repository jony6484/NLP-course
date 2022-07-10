def foo_fn(text):
    """
    Some documentation.
    Use This function wisely
    """
    words = []
    for word in text.split(" "):
        words.append(word)
    return words


def foo_2(text):
    letters = foo_fn(text=text)
    print(letters)
