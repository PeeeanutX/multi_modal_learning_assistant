import re


def clean_text(text):
    text = re.sub('\s+', ' ', text)
    text = re.sub('[^A-Za-z0-9 .,;:!?()-]', '', text)
    return text.strip()

