import spacy
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer

regexp = re.compile('(?u)\\b\\w\\w+\\b')

en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string)
)

stemmer = nltk.stem.PorterStemmer()

def compare_normalization(doc):
    doc_spacy = en_nlp(doc)

    print("Lemma: \n{}".format([token.lemma_ for token in doc_spacy]))
    print("Stem: \n{}".format([stemmer.stem(token.norm_.lower()) for token in doc_spacy]))

def custom_tokenizer(document):
    doc_spacy = en_nlp(document, entity=False, parse=False)

    return [token.lemma_ for token in doc_spacy]

lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)
