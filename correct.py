"""
	Spell-Check & Correction Program (word-level)

	Rahul Kejriwal, CS14B023
	Srinidhi Prabhu, CS14B028
"""

import math
import csv
import re
import cPickle
import numpy as np

import fuzzy
from weighted_levenshtein import dam_lev


class SpellChecker:

	def __init__(self, word_set, unigrams, k, edit_counts, lamda=1, alphabet='abcdefghijklmnopqrstuvwxyz'):

		# Initialize alphabet
		self.alphabet = alphabet

		# Store all known words
		self.dict_words = word_set

		# Weighting likelihood & prior
		self.lamda = lamda

		# Store unigram probabilities - Use Laplace Add-k Smoothing for log probabilities
		self.priors = {}
		self.k = k
		self.N = sum((count for word, count in unigrams)) + k*len(unigrams) + k
		for word, count in unigrams:
			self.priors[word] = math.log(float(count+k) / self.N)

		# Edit Distance Costs
		self.insert_costs = np.ones((128,))
		self.delete_costs = np.ones((128,))
		self.transpose_costs  = np.ones((128,128))
		self.substitute_costs = np.ones((128,128))
		
		"""
		# Store edit counts
		self.edit_counts = {}
		for edit, count in edit_counts:
			self.edit_counts[edit] = count + k
		self.total_edits = sum((count for edit,count in edit_counts)) + k*len(edit_counts) + k
		"""

		# Build phonetic index - Double Metaphone
		self.dmeta = fuzzy.DMetaphone()
		self.phonetic_buckets = {}

		for word in self.dict_words:
			phonetic_idx = self.dmeta(word)

			if phonetic_idx[0] not in self.phonetic_buckets:
				self.phonetic_buckets[phonetic_idx[0]] = []
			self.phonetic_buckets[phonetic_idx[0]].append(word)

			if phonetic_idx[1] != None:
				if phonetic_idx[1] not in self.phonetic_buckets:
					self.phonetic_buckets[phonetic_idx[1]] = []
				self.phonetic_buckets[phonetic_idx[1]].append(word)


	def __edit_neighbors_1(self, word):
		word_len = len(word)
		deletions  		= [(word[:i]+word[i+1:]) for i in range(word_len)]
		insertions 		= [word[:i]+letter+word[i:] for i in range(word_len+1) for letter in self.alphabet]
		substitutions  	= [word[:i]+letter+word[i+1:] for i in range(word_len) for letter in self.alphabet]
		transpositions 	= [word[:i]+word[i+1]+word[i]+word[i+2:] for i in range(word_len-1)]
		return set(deletions + insertions + substitutions + transpositions)


	def __filter_unknown(self, words):
		return set([word for word in words if word in self.dict_words])


	def __generateCandidates(self, wrong_word):
		candidates = self.__edit_neighbors_1(wrong_word)
		candidates = self.__filter_unknown(candidates)
		
		candidates_2 = set([next_candidate for candidate in candidates for next_candidate in self.__edit_neighbors_1(candidate)])
		candidates_2 = self.__filter_unknown(candidates_2)

		metaphone_bkts = self.dmeta(wrong_word)
		candidates_3 = self.phonetic_buckets[metaphone_bkts[0]] + (self.phonetic_buckets[metaphone_bkts[1]] if metaphone_bkts[1] != None else [])
		candidates_3 = set(candidates_3)

		return (candidates_3.union(candidates).union(candidates_2))


	def __score(self, wrong_word, candidate):
		dl_dist = dam_lev(wrong_word, candidate, insert_costs=self.insert_costs, substitute_costs=self.substitute_costs, delete_costs=self.delete_costs, transpose_costs=self.transpose_costs)
		log_prior = self.priors[candidate] if candidate in self.priors else (float(self.k) / self.N)
		return -dl_dist + self.lamda * log_prior


	def __rankCandidates(self, wrong_word, candidates):
		return [(candidate, self.__score(wrong_word, candidate)) for candidate in candidates]


	def correct(self, wrong_word, top_k=3):
		candidates = self.__generateCandidates(wrong_word)
		scores	   = self.__rankCandidates(wrong_word, candidates)
		return sorted(scores, key= lambda x:-x[1])[:top_k]


	"""
	def __prob_edit(self, edits):


	def __recursiveCandidates(self, head, tail, edits_left, edits_used, results):
		curr_state = head + tail
		
		if curr_state in self.dict_words:
			if curr_state in results:	results[curr_state] = max(results[curr_state], edits_used, key=self.__prob_edit)
			else:						results[curr_state] = edits_used

		if edits_left <= 0:		
			return

		possible_extensions = [head + c for c in self.alphabet]


	def __generateCandidates(self, wrong_word):
		results = {}
		self.__recursiveCandidates('', wrong_word, 2, [], results)
		return results
	"""


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


"""
	Reads unigrams and corresponding frequency counts
"""
def read_unigram_probs(unigram_file):
	with open(unigram_file) as fp:
		lines = [[tok.strip() if i==0 else int(tok.strip()) for i, tok in enumerate(line.split('\t'))] for line in fp]
	return lines


"""
	Read letter edit counts
"""
def read_edit_counts(edit_file):
	with open(edit_file) as fp:
		lines = [[el if i==0 else int(el.strip()) for i, el in enumerate(line.split('\t'))] for line in fp]
		# lines = [[el if i==0 else int(el.strip()) for i, tok in enumerate(line.split('\t')) for el in tok.split('|')] for line in fp]
	return lines


if __name__ == '__main__':

	first = True

	# If executing first time
	if first:

		# Read dictionaries for candidate generation
		word_set = read_csv_dict('Data/Dictionaries/dictionary.csv')
		word_set = word_set.union(read_list_dict('Data/Dictionaries/word.list'))

		# Read unigram counts for prior/LM model
		unigrams = read_unigram_probs('Data/count_1w.txt') 

		# Read edit counts for likelihood/channel model
		edit_counts = read_edit_counts('Data/count_1edit.txt')

		# Build Checker model
		checker = SpellChecker(word_set, unigrams, 1, edit_counts, lamda=0.05)
		with open('model.pkl', 'wb') as fp:
			cPickle.dump(checker, fp)

	# Load Checker model
	with open('model.pkl', 'rb') as fp:
		checker = cPickle.load(fp)

	print checker.correct('emberassment')