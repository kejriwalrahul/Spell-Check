from correct import *
from sentences import *
import operator

class PhraseChecker(SpellChecker):
	
	def __init__(self,word_set, unigrams, k, unigram_file, bigram_file, homophones, costs=None, lamda=1, alphabet='abcdefghijklmnopqrstuvwxyz'):
		SpellChecker.__init__(self,word_set, unigrams, k, costs, lamda)
		
		self.homophones = homophones
		
		fp = open(bigram_file,'r')
		
		self.next_word = {}
		
		self.prev_word = {}
		
		for line in fp:
			line = line.strip().split()
			word1 = line[0].lower()
			word2 = line[1].lower()
			val = int(line[2])
			
			if word1 in self.next_word:
				self.next_word[word1].append((word2,val))
			else:
				self.next_word[word1] = [(word2,val)]
				
			if word2 in self.prev_word:
				self.prev_word[word2].append((word1,val))
			else:
				self.prev_word[word2] = [(word1,val)]
			
			
		fp.close()
		
		fp = open(unigram_file,'r')
		
		self.unigram_count = {}
		for line in fp:
			line = line.strip().split()
			self.unigram_count[line[0].lower()] = int(line[1])
			
		self.dict_words = self.dict_words.union(set(self.unigram_count.keys()))
		
		
		
	def correct(self,words,alpha=0.01,top_k=3):
		
		all_in_dict = True
		isHomophonic = False
		
		index_wrong = -1
		n = len(words)
		
		for i,word in enumerate(words):
			if word not in self.dict_words:
				all_in_dict = False
				index_wrong = i
				break
				
		if all_in_dict:
		
			homophonic = []
			for i,word in enumerate(words):
				if word in self.homophones and word not in stops:
					homophonic.append((i,word))
					
			if len(homophonic) > 0:
				for i in range(0,len(homophonic)):
					candidates = self.homophones[homophonic[i][1]]
					
					possible = []
					for candidate in candidates:
						ind = homophonic[i][0]
					
						mult = False
						val = 0
						if ind != 0:
							prev = words[ind-1]
							if candidate in self.prev_word:
								for item in self.prev_word[candidate]:
									if prev == item[0]:
										val += item[1]
										break
							
						else:
							mult = True
				
						if ind != n-1:
							next = words[ind+1]
							if candidate in self.next_word:
								for item in self.next_word[candidate]:
									if next == item[0]:
										val += item[1]
										break
							
						else:
							mult = True
						
						if mult:
							val *= 2
							
						if candidate in self.unigram_count:
							if not mult:
								val/=(float(self.unigram_count[words[ind-1]]+self.unigram_count[words[ind+1]]))
							else:
								if ind == 0:
									val/=(float(2*self.unigram_count[words[ind+1]]))
								else:
									val/=(float(2*self.unigram_count[words[ind-1]]))
									
						else:
							continue
						
						if val > 1e-5:
							possible.append((homophonic[i][0],candidate,val))
							
				if len(possible) > 0:
					curr_tuple = possible[0]
					for i in range(1,len(possible)):
						if possible[i][2] > curr_tuple[2]:
							curr_tuple = possible[i]
					
					index_wrong = curr_tuple[0]
					isHomophonic = True
					h_words = [curr_tuple[1]]			
							
					
			if index_wrong == -1:				
				min_val = float("inf")
				for i in range(0,n):
					mult = False
					val = 0
					if i != 0:
						prev = words[i-1]
						if words[i] in self.prev_word:
							for item in self.prev_word[words[i]]:
								if prev == item[0]:
									val += item[1]
									break
							
					else:
						mult = True
				
					if i != n-1:
						next = words[i+1]
						if words[i] in self.next_word:
							for item in self.next_word[words[i]]:
								if next == item[0]:
									val += item[1]
									break
							
					else:
						mult = True
			
					if mult:
						val *= 2
				
					# Add the weighted unigram count as well
			
					if words[i] in self.unigram_count:
						val += alpha*self.unigram_count[words[i]]
				
					if val < min_val:
						min_val = val
						index_wrong = i
								
			
		
		
		word = words[index_wrong]
		candidates = list(self.generateCandidates(word))
		
		dict_of_candidates = {}
		
		
		for candidate in candidates:
		
			#print candidate
			if candidate == word:
				continue
			val = 0
			mult = False
			if index_wrong > 0:
				prev = words[index_wrong-1]
				if candidate in self.prev_word:
					for item in self.prev_word[candidate]:
						if prev == item[0]:
							val += item[1]
							break
			else:
				mult = True
				
			if index_wrong < n-1:
				next = words[index_wrong+1]
				if candidate in self.next_word:
					for item in self.next_word[candidate]:
						if next == item[0]:
							val += item[1]
							break
			else:
				mult = True
				
			if mult:
				val *= 2
				
			if words[i] in self.unigram_count:
				val += alpha*self.unigram_count[words[i]]
				
			dist = dam_lev(word, candidate, insert_costs=self.insert_costs, substitute_costs=self.substitute_costs, delete_costs=self.delete_costs, transpose_costs=self.transpose_costs)
			
			dict_of_candidates[candidate] = val/(np.exp(dist))
			
			
		sorted_dict = sorted(dict_of_candidates.items(), key=operator.itemgetter(1))
		
		
		sorted_dict.reverse()		
		
		corrected = sorted_dict[:top_k]
		
		correct_words = [item[0] for item in corrected]
		
		if isHomophonic:
			correct_words = h_words + correct_words
			
		correct_words = correct_words[:top_k]
		
		
		return (correct_words,words[index_wrong],[item[1] for item in corrected])
		
			
			
			
			
if __name__ == '__main__':

	if len(sys.argv) != 3:
		print "Usage: python phrases.py <infile> <outfile>"
		sys.exit(1)
	
	FRESH = False
	DEBUG = False
	
	if FRESH:

		# Read dictionaries for candidate generation
		word_set = read_csv_dict('Data/Dictionaries/dictionary.csv')
		word_set = word_set.union(read_list_dict('Data/Dictionaries/word.list'))

		# Read unigram counts for prior/LM model
		unigrams = read_unigram_probs('Data/count_1w.txt') 

		
		vec_file = 'Data/Vectors/glove.6B.50d.txt'
		
		bigram_file = 'Data/Dictionaries/bigrams.txt'
		
		unigram_file = 'Data/Dictionaries/unigrams.txt'
		
		homophones = get_homophones_from_file('Data/Dictionaries/homophones.txt')
		
		# Build Checker model
		p_checker = PhraseChecker(word_set, unigrams, 1, unigram_file, bigram_file, homophones, lamda=0.05)
		
		with open('Data/Models/phrase_model.pkl', 'wb') as fp:
			cPickle.dump(p_checker, fp)
			
	
	# Load Checker model
	
	with open('Data/Models/phrase_model.pkl', 'rb') as fp:
		p_checker = cPickle.load(fp)
	
	
	# For every sentence, for each word - generate candidates, take context words, form representations and then find scores for each candidate(upto 3, say)
	
	
	# Output results
	stops = set(stopwords.words('english'))
	with open(sys.argv[1]) as fin, open(sys.argv[2], 'w') as fout:
		for line in fin:
			sentence = line.strip().lower()
			sentence = sentence.replace('.','')
			
			words = sentence.strip().split()
			
			suggestions,wrong,scores = p_checker.correct(words,alpha=0.001,top_k=3)
			if DEBUG:
				fout.write('\t'.join([wrong] + [word for word in suggestions]) + '\n')
				fout.write('\t'.join([str(score) for score in scores])+'\n')
			else:
				fout.write('\t'.join([wrong] + [word for word in suggestions]) + '\n')
				
	
