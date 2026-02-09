import numpy as np

class NaiveBayesDetector:
    def __init__(self):
        #dictionaries to store word counts
        self.spam_counts = {}
        self.ham_counts = {}
        #probabilities of spam and ham(priors)
        self.p_spam = 0
        self.p_ham = 0
        #totals for the formula
        self.vocab_size = 0
        self.n_spam_words = 0
        self.n_ham_words = 0

    def fit(self, messages, labels):
        """

        The Training part: The model studies the word frequencies of each message
        """
        total_messages = len(labels)
        self.p_spam = sum(labels) / total_messages
        self.p_ham = 1 - self.p_spam

        unique_words = set()

        for msg, label in zip(messages, labels):
            for word in msg:
                unique_words.add(word)
                if label == 1:
                    self.spam_counts[word] = self.spam_counts.get(word, 0) + 1
                    self.n_spam_words += 1
                else:
                    self.ham_counts[word] = self.ham_counts.get(word, 0) + 1
                    self.n_ham_words += 1

        self.vocab_size = len(unique_words)

    def predict(self, message_tokens):
        """

        :param message_tokens:
        :return:
        """

        spam_score = np.log(self.p_spam)
        ham_score = np.log(self.p_ham)

        for word in message_tokens:
            p_word_spam = (self.spam_counts.get(word, 0) + 1) / (self.n_spam_words + self.vocab_size)
            p_word_ham = (self.ham_counts.get(word, 0) + 1) / (self.n_ham_words + self.vocab_size)

            spam_score += np.log(p_word_spam)
            ham_score += np.log(p_word_ham)
        return 1 if spam_score > ham_score else 0