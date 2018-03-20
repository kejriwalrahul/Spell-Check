"""
	Spell-Check & Correction Program (word-level)

	Rahul Kejriwal, CS14B023
	Srinidhi Prabhu, CS14B028
"""

import math
import csv
import sys
import re
import cPickle
import numpy as np

import fuzzy
from weighted_levenshtein import dam_lev
from tqdm import tqdm

class SpellChecker:

	def __init__(self, word_set, unigrams, k, costs=None, lamda=1, alphabet='abcdefghijklmnopqrstuvwxyz'):

		# Initialize alphabet
		self.alphabet = alphabet

		# Store all known words
		self.dict_words = word_set

		# Build and store valid prefixes
		self.valid_prefixes = set([])
		for word in self.dict_words:
			for i in range(len(word)+1):
				self.valid_prefixes.add(word[:i])

		# Weighting likelihood & prior
		self.lamda = lamda

		# Store unigram probabilities - Use Laplace Add-k Smoothing for log probabilities
		self.priors = {}
		self.k = k
		self.N = sum((count for word, count in unigrams)) + k*len(unigrams) + k
		for word, count in unigrams:
			self.priors[word] = math.log(float(count+k) / self.N)

		# Edit Distance Costs
		if costs != None:
			self.insert_costs = costs['ins_costs']
			self.delete_costs = costs['del_costs']
			self.substitute_costs = costs['sub_costs']
			self.transpose_costs  = costs['trans_costs']
		else:
			self.insert_costs = np.ones((128,))
			self.delete_costs = np.ones((128,))
			self.transpose_costs  = np.ones((128,128))
			self.substitute_costs = np.ones((128,128))
		
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


	def __fastGenerateNeighbors(self, left, right, max_dist=2):
		# Boundary Conditions
		if max_dist == 0:
			if left+right in self.valid_prefixes:	return [left+right]
			else:									return []

		if len(right) == 0:
			results = []
			if left in self.valid_prefixes:
				results.append(left)
			for letter in self.alphabet:
				if left + letter in self.valid_prefixes:
					results.append(left + letter)	
			return list(set(results))

		# Update bounds
		left = left + right[:1]
		right = right[1:]

		# Initialize neighbors
		neighbor_set = []

		# Deletions
		if left[:-1] in self.valid_prefixes:
			neighbor_set += self.__fastGenerateNeighbors(left[:-1], right, max_dist-1)

		# Insertions	
		for letter in self.alphabet:
			if left[:-1]+letter+left[-1:]  in self.valid_prefixes:
				neighbor_set += self.__fastGenerateNeighbors(left[:-1]+letter+left[-1:], right, max_dist-1)

		# Substitutions
		for letter in self.alphabet:
			if left[:-1]+letter in self.valid_prefixes:
				neighbor_set += self.__fastGenerateNeighbors(left[:-1]+letter, right, max_dist - (1 if letter != left[-1] else 0))

		# Transpositions
		if len(right) >= 1:
			if left[:-1] + right[0] + left[-1] in self.valid_prefixes:
				neighbor_set += self.__fastGenerateNeighbors(left[:-1]+right[0]+left[-1], right[1:], max_dist-1)

		return list(set(neighbor_set))


	def __generateCandidates(self, wrong_word):
		"""
		# Old Approach - Too Slow (remove candidates_2 for fast+efficient)
		candidates = self.__edit_neighbors_1(wrong_word)		
		candidates_2 = set([next_candidate for candidate in candidates for next_candidate in self.__edit_neighbors_1(candidate)])
		
		candidates = self.__filter_unknown(candidates)
		candidates_2 = self.__filter_unknown(candidates_2)
		"""

		# Edit Distance based candidates
		candidates = self.__fastGenerateNeighbors('', wrong_word, 2)
		candidates = self.__filter_unknown(candidates)

		# DMetaphone based candidates
		metaphone_bkts = self.dmeta(wrong_word)
		candidates_meta = self.phonetic_buckets.get(metaphone_bkts[0], []) + (self.phonetic_buckets.get(metaphone_bkts[1], []) if metaphone_bkts[1] != None else [])
		candidates_meta = set(candidates_meta)

		return (candidates_meta.union(candidates))

		
	def generateCandidates(self,wrong_word):
		return self.__generateCandidates(wrong_word)


	def __score(self, wrong_word, candidate):
		dl_dist = dam_lev(wrong_word, candidate, insert_costs=self.insert_costs, substitute_costs=self.substitute_costs, delete_costs=self.delete_costs, transpose_costs=self.transpose_costs) / max(len(wrong_word), len(candidate))
		log_prior = self.priors[candidate] if candidate in self.priors else math.log(float(self.k) / self.N)
		return -dl_dist + self.lamda * log_prior


	def __rankCandidates(self, wrong_word, candidates):
		return [(candidate, self.__score(wrong_word, candidate)) for candidate in candidates]


	def correct(self, wrong_word, top_k=3):
		candidates = self.__generateCandidates(wrong_word)
		scores	   = self.__rankCandidates(wrong_word, candidates)
		return sorted(scores, key= lambda x:-x[1])[:top_k]


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
	Check accuracy of model and compare with other libs
"""
def error_file_accuracy(file, checker, fil_type=0, verbose=False, suppress=False):

	# Read file
	with open(file) as fp:
		lines = ''.join(fp.readlines())

	# Parse cases
	ws = []
	cs = []
	if fil_type == 0:
		instances = lines.split('$')
		for instance in instances:
			toks = [el for el in instance.split('\n') if el != '']
			ws += toks[1:]
			cs += [toks[0] for i in range(len(toks)-1)]
	elif fil_type == 1:
		instances = lines.split('\n')
		for instance in instances:
			toks = [el for el in instance.split(':') if el != '']
			curr_ws = [el.split('*')[0].strip() for el in toks[1].split(',')]
			ws += curr_ws
			cs += [toks[0].strip() for i in range(len(curr_ws))]
	elif fil_type == 2:
		instances = lines.split('$')
		for instance in instances:
			toks = [el for el in instance.split('\n') if el != '']
			ws += [tok.split(' ')[0] for tok in toks[1:]]
			cs += [toks[0] for i in range(len(toks)-1)]		
	else:
		raise NotImplementedError

	"""
		Get model score
	"""
	score = 0.0
	for w,c in tqdm(zip(ws, cs)):
		guesses = checker.correct(w, 10)
		try: 
			curr_mrr = 1.0 / (1 + next(i for i, (guess,score) in enumerate(guesses) if guess==c))
		except:
			curr_mrr = 0.0
		score += curr_mrr
		if verbose:
			print "(%s,%s): %s" % (w,c,str(guesses))
	score = score / len(ws)

	if not suppress:
		"""
			Compare with PyEnchant Lib
		"""	
		import enchant
		d = enchant.Dict('en_GB')
		score2 = 0.0
		for w,c in tqdm(zip(ws, cs)):
			guesses = d.suggest(w)
			try: 
				curr_mrr = 1.0 / (1 + next(i for i, guess in enumerate(guesses) if guess==c))
			except:
				curr_mrr = 0.0
			score2 += curr_mrr
		score2 = score2 / len(ws)

		"""
			Compare with Autocorrect Lib
		"""
		from autocorrect import spell
		score3 = 0.0
		for w,c in tqdm(zip(ws, cs)):
			if c == spell(w):
				score3 += 1.0
		score3 = score3 / len(ws)

		return score, score2, score3
	else:
		return score


if __name__ == '__main__':

	# cmdline args check
	if len(sys.argv) != 3:
		print "Usage: python correct.py <infile> <outfile>"
		sys.exit(1)

	FRESH = False
	DEBUG = False

	# If executing first time
	if FRESH:

		"""
			# Deprecated Dictionaries
			word_set = read_csv_dict('Data/Dictionaries/dictionary.csv')
			word_set = word_set.union(read_list_dict('Data/Dictionaries/word.list'))
		"""

		# Read dictionaries for candidate generation
		word_set = read_list_dict('Data/Dictionaries/correct_exp.list')

		# Read unigram counts for prior/LM model
		unigrams = read_unigram_probs('Data/count_1w.txt') 

		# Read edit costs
		with open("Data/costs.npz") as fp:
			costs = np.load(fp)
			costs = {
				'ins_costs': costs['ins_costs'],
				'del_costs': costs['del_costs'],
				'sub_costs': costs['sub_costs'],
				'trans_costs': costs['trans_costs'],
			}

		# Build Checker model
		checker = SpellChecker(word_set, unigrams, 1, costs=costs, lamda=0.05)
		with open('Data/Models/model.pkl', 'wb') as fp:
			cPickle.dump(checker, fp)

	# Load Checker model
	with open('Data/Models/model.pkl', 'rb') as fp:
		checker = cPickle.load(fp)

	# Output results
	with open(sys.argv[1]) as fin, open(sys.argv[2], 'w') as fout:
		for line in fin:
			word = line.strip()
			guesses = checker.correct(word, 10)
			if DEBUG == True:
				fout.write('\t'.join([word] + ['(%s,%.2f)'%(guess,score) for guess,score in guesses]) + '\n')
			else:
				fout.write('\t'.join([word] + [guess for guess, score in guesses]) + '\n')

	# Measure model accuracy
	# suppress = True
	# print "Accuracy: ", error_file_accuracy('Data/Errors/word_val_set.dat', checker, fil_type=0, suppress=suppress)
	# print "Accuracy: ", error_file_accuracy('Data/Errors/missp.dat', checker, fil_type=0, suppress=suppress)
	# print "Accuracy: ", error_file_accuracy('Data/Errors/aspell.dat', checker, fil_type=0, suppress=suppress)
	# print "Accuracy: ", error_file_accuracy('Data/Errors/wikipedia.dat', checker, fil_type=0, suppress=suppress)
	# print "Accuracy: ", error_file_accuracy('Data/Errors/spell-errors.txt', checker, fil_type=1, suppress=suppress)
	# print "Accuracy: ", error_file_accuracy('Data/Errors/holbrook-missp.dat', checker, fil_type=2, suppress=suppress)