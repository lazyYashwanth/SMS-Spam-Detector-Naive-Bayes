import string

def clean_text(message):
    message = str(message).lower()

    clean_message = "".join([char for char in message if char not in string.punctuation])

    words = clean_message.split()

    return words