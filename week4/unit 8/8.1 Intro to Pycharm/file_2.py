def foo_fn(text):
    letters_list = []
    for word in text.split(" "):
        for letter in word:
            letters_list.append(letter)
    return letters_list


def foo_2(text):
    letters = foo_fn(text)
    print(letters)
