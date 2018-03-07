"""
	Spell-Check & Correction Program (word-level)

	Rahul Kejriwal, CS14B023
	Srinidhi Prabhu, CS14B028
"""

import csv
import re
import cPickle
import fuzzy

class SpellChecker:

	def __init__(self, word_set):

		# Store all known words
		self.dict_words = word_set

		# Build phonetic index - Double Metaphone
		self.dmeta = fuzzy.DMetaphone()
		self.phonetic_buckets = {}

		for word in self.dict_words:
			phonetic_idx = self.dmeta(word)

			if phonetic_idx[0] not in self.phonetic_buckets:
				self.phonetic_buckets[phonetic_idx[0]] = []
			self.phonetic_buckets[phonetic_idx[0]].append(word)

			if phonetic_idx[1] not in self.phonetic_buckets:
				self.phonetic_buckets[phonetic_idx[1]] = []
			self.phonetic_buckets[phonetic_idx[1]].append(word)


	def __edit_neighbors_1(self, word):
		word_len = len(word)
		deletions  		= [word[:i]+word[i+1:] for i in range(word_len)]
		insertions 		= [word[:i]+letter+word[i:] for i in range(word_len+1) for letter in 'abcdefghijklmnopqrstuvwxyz']
		substitutions  	= [word[:i]+letter+word[i+1:] for i in range(word_len) for letter in 'abcdefghijklmnopqrstuvwxyz']
		transpositions 	= [word[:i]+word[i+1]+word[i]+word[i+2:] for i in range(word_len-1)]
		return set(deletions + insertions + substitutions + transpositions)

	def __filter_unknown(self, words):
		return set([word for word in words if word in self.dict_words])

	def generateCandidates(self, wrong_word):
		candidates = self.__edit_neighbors_1(wrong_word)
		candidates_2 = set([next_candidate for candidate in candidates for next_candidate in self.__edit_neighbors_1(candidate)])
		candidates = self.__filter_unknown(candidates)
		candidates_2 = self.__filter_unknown(candidates_2)
		print candidates
		print candidates_2

		candidates_3 = self.phonetic_buckets[self.dmeta(wrong_word)[0]]
		print sorted(candidates_3)


"""
	Extract word set from csv file
"""
def read_csv_dict(dictfile):
	with open(dictfile) as fp:
		csv_reader = csv.reader(fp)
		dict_words = set([word.replace("\"","").strip().lower() for word, _, _ in csv_reader])
	return dict_words


"""
	Read word list file 
"""
def read_list_dict(dictfile):
	with open(dictfile) as fp:
		words = set([line.strip() for line in fp])
	return words


if __name__ == '__main__':

	first = False

	# If executing first time
	if first:
		word_set = read_csv_dict('Data/Dictionaries/dictionary.csv')
		word_set = word_set.union(read_list_dict('Data/Dictionaries/word.list'))

		# Build Checker model
		checker = SpellChecker(word_set)
		with open('model.pkl', 'wb') as fp:
			cPickle.dump(checker, fp)

	# Load Checker model
	with open('model.pkl', 'rb') as fp:
		checker = cPickle.load(fp)

	checker.generateCandidates('emberassment')