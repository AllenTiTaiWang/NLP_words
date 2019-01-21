from typing import List, Text, Tuple
import numpy as np
from collections import Counter
import re
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from operator import itemgetter

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

    with open(word_pos_path) as f:
        data = f.read()
        line = data.split()

        if pos_regex is not None and word_regex is None:
            line = [item.rsplit('/', 1)[1] for item in line]
            line = [item for item in line if re.match(pos_regex, item) is not  None]
        
        elif pos_regex is None and word_regex is not None:
            line = [item.rsplit('/', 1)[0] for item in line]
            line = [item for item in line if re.match(word_regex, item) is not  None]

        else:
            line = [item for item in line if re.match(word_regex, item.rsplit('/', 1)[0]) is not  None]
            line = [item for item in line if re.match(pos_regex, item.rsplit('/', 1)[1]) is not  None]
 
        common = Counter(line).most_common(n)
        return common

class WordVectors(object):
    def __init__(self, word_vectors_path: Text):
        """Reads words and their vectors from a file.

        :param word_vectors_path: The path of a file containing word vectors.
        Each line should be formatted as a single word, followed by a
        space-separated list of floating point numbers. For example:

            the 0.063380 -0.146809 0.110004 -0.012050 -0.045637 -0.022240
        """
      
        self.dic = {}
        with open(word_vectors_path) as f:
            for line in f:
                word = line.split(' ', 1)[0]
                vec = line.split(' ', 1)[1].strip()
                vec_a = np.fromstring(vec, sep = " ")
                self.dic[word] = vec_a

    def average_vector(self, words: List[Text]) -> np.ndarray:
        """Calculates the element-wise average of the vectors for the given
        words.

        For example, if the words correspond to the vectors [1, 2, 3] and
        [3, 4, 5], then the element-wise average should be [2, 3, 4].

        :param words: The words whose vectors should be looked up and averaged.
        :return: The element-wise average of the word vectors.
        """

        words_v = [self.dic[item] for item in words]
        vec = np.average(words_v, axis = 0)
        return vec

    def most_similar(self, word: Text, n=10) -> List[Tuple[Text, int]]:
        """Finds the most similar words to a query word. Similarity is measured
        by cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
        over the word vectors.

        :param word: The query word.
        :param n: The number of most similar words to return.
        :return: The n most similar words to the query word.
        """
        #heapq.nlargest
        #scikit-learn
        word_vec = self.dic[word]
        cs_list = []
        for key, value in self.dic.items():
            cs = cosine_similarity(word_vec.reshape(1, -1), value.reshape(1, -1))
            cs_list.append((key, float(cs)))
        result = heapq.nlargest(n+1, cs_list, key = itemgetter(1))[1:]
        return result
