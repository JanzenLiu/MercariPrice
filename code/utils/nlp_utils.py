import nltk


# nltk.download('popular') # should be run while using nltk for the first time


def tokenize_doc(doc):
    sents = nltk.sent_tokenize(doc)  # list of sentences, each item is a string standing for a sentence
    return [nltk.word_tokenize(sent) for sent in sents]  # each item is a list containing words in the sentence
