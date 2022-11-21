# A class that is used to convert a collection of text documents to a matrix of token counts.
from classification.utils.utility import *
from sklearn.feature_extraction.text import CountVectorizer


class NgramsExtractor:
    def __init__(self, max_ngram_len=2):
        self.packet_counter = CountVectorizer(analyzer='word',
                                              tokenizer=lambda x: x.split(),
                                              stop_words=None,
                                              ngram_range=(1, max_ngram_len), )
        self.burst_counter = CountVectorizer(analyzer='word',
                                             tokenizer=lambda x: x.split(),
                                             stop_words=None,
                                             ngram_range=(1, max_ngram_len), )

    def fit(self, x, y=None):
        bursts = x.lengths.apply(get_bursts)
        self.packet_counter.fit(x.lengths.apply(join_str))
        self.burst_counter.fit(bursts.apply(join_str))
        return self

    def transform(self, data_list):
        bursts = data_list.lengths.apply(get_bursts)
        data_str = data_list.lengths.apply(join_str)
        bursts_str = bursts.apply(join_str)

        packet_ngrams = self.packet_counter.transform(data_str)
        burst_ngrams = self.burst_counter.transform(bursts_str)

        return np.concatenate((packet_ngrams.todense(),
                               burst_ngrams.todense()), axis=1)
