from typing import List, Text, Tuple
import numpy as np


def most_common(word_pos_path: Text,
                word_regex=".*",
                pos_regex=".*",
                n=10) -> List[Tuple[Text, int]]:
    """Finds the most common words and/or parts of speech in a file.

    :param word_pos_path: The path of a file containing part-of-speech tagged
    text. The file should be formatted as a sequence of tokens separated by
    whitespace. Each token should be a word and a part-of-speech tag, separated
    by a slash. For example: "The/at Hartsfield/np home/nr is/bez at/in 637/cd
    E./np Pelham/np Rd./nn-tl Aj/nn ./."

    :param word_regex: If None, do not include words in the output. If a regular
    expression string, all words included in the output must match the regular
    expression.

    :param pos_regex: If None, do not include part-of-speech tags in the output.
    If a regular expression string, all part-of-speech tags included in the
    output must match the regular expression.

    :param n: The number of most common words and/or parts of speech to return.

    :return: A list of (token, count) tuples for the most frequent words and/or
    parts-of-speech in the file. Note that, depending on word_regex and
    pos_regex (as described above), the returned tokens will contain either
    words, part-of-speech tags, or both.
    """


class WordVectors(object):
    def __init__(self, word_vectors_path: Text):
        """Reads words and their vectors from a file.

        :param word_vectors_path: The path of a file containing word vectors.
        Each line should be formatted as a single word, followed by a
        space-separated list of floating point numbers. For example:

            the 0.063380 -0.146809 0.110004 -0.012050 -0.045637 -0.022240
        """

    def average_vector(self, words: List[Text]) -> np.ndarray:
        """Calculates the element-wise average of the vectors for the given
        words.

        For example, if the words correspond to the vectors [1, 2, 3] and
        [3, 4, 5], then the element-wise average should be [2, 3, 4].

        :param words: The words whose vectors should be looked up and averaged.
        :return: The element-wise average of the word vectors.
        """

    def most_similar(self, word: Text, n=10) -> List[Tuple[Text, int]]:
        """Finds the most similar words to a query word. Similarity is measured
        by cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
        over the word vectors.

        :param word: The query word.
        :param n: The number of most similar words to return.
        :return: The n most similar words to the query word.
        """
