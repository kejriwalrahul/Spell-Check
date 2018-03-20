from correct import *
from nltk.corpus import stopwords
class SentenceChecker(SpellChecker):
	
	def __init__(self,word_set, unigrams, k, vector_file, homophones, costs=None, lamda=1, alphabet='abcdefghijklmnopqrstuvwxyz'):
		SpellChecker.__init__(self,word_set, unigrams, k, costs, lamda)
		
		self.homophones = homophones
		
		# Get the preprocessed word vectors
		
		with open(vector_file, 'r') as f:
			words = [x.rstrip().split(' ')[0] for x in f.readlines()]
		with open(vector_file, 'r') as f:
			vectors = {}
			for line in f:
				vals = line.rstrip().split(' ')
				vectors[vals[0]] = [float(x) for x in vals[1:]]

		vocab_size = len(words)
		self.vocab = {w: idx for idx, w in enumerate(words)}
		self.ivocab = {idx: w for idx, w in enumerate(words)}

		vector_dim = len(vectors[self.ivocab[0]])
		W = np.zeros((vocab_size, vector_dim))
		for word, v in vectors.items():
			if word == '<unk>':
				continue
			W[self.vocab[word], :] = v

		# normalize each word vector to unit variance
		W_norm = np.zeros(W.shape)
		d = (np.sum(W ** 2, 1) ** (0.5))
		self.W_norm = (W.T / d).T
		
	def generateNonStopCandidates(self,word,stops):
		return self.generateCandidates(word).difference(set(stops))
		
	def get_vector(self,word_list):
		count = 0
		for i,term in enumerate(word_list):
			if count == 0:
				try:
					vec_result = np.copy(self.W_norm[self.vocab[term], :])
					count += 1
				except:
					pass
			else:
				try:
					vec_result += self.W_norm[self.vocab[term], :]
					count += 1
				except:
					pass
		
		try:	
			vec_result = vec_result/float(count)
		except:
			vec_result = np.zeros(self.W_norm.shape[1])
		
		vec_norm = np.zeros(vec_result.shape)
		d = (np.sum(vec_result ** 2,) ** (0.5))
		try:
			vec_norm = (vec_result.T / d).T
		except:
			vec_norm = 0
		
		return vec_norm
		
	def get_all_vectors(self,wrong,words):
		
		word_vals = [term for term in words if term in self.vocab]
		word_vec = np.array([self.W_norm[self.vocab[term], :] for term in words if term in self.vocab])
		word_dist = np.array([dam_lev(wrong, term, insert_costs=self.insert_costs, substitute_costs=self.substitute_costs, delete_costs=self.delete_costs, transpose_costs=self.transpose_costs) for term in words if term in self.vocab])
		
		return (word_vals,word_vec,word_dist)
		
	
	# for each word - generate candidates, take context words, form representations and then find scores for each candidate(upto 3, say)
	# add weights to the dam_lau edit distance
	def correct(self, words, stops, alpha=1,top_k=3):
	
		all_in_dict = True
		all_in_glove = True
		isHomophonic = False
		
		
		for i,word in enumerate(words):
			if word not in self.dict_words:
				all_in_dict = False
				index_wrong = i
				break
				
		if all_in_dict:
		
			for i,word in enumerate(words):
				if word not in self.vocab:
					all_in_glove = False
					index_wrong = i
					break
					
			if all_in_glove:
				homophonic = []
				for i,word in enumerate(words):
					if word in self.homophones and word not in stops:
						homophonic.append((i,word))
					
				if len(homophonic) == 0:
					# Need a better method for this			
					cos_sim = 2
					index_wrong = -1
					for i,word in enumerate(words):
						if word not in stops:
							v1 = self.get_vector([word])
							v2 = self.get_vector(get_context_words(i,words))
							sim = np.dot(v1,v2)
							if sim < cos_sim:
								cos_sim = sim
								index_wrong = i
							
				elif len(homophonic) == 1:
					isHomophonic = True
					index_wrong = homophonic[0][0]
					h_words = self.homophones[homophonic[0][1]]
			
				else:
					isHomophonic = True
					cos_sim = 2
					index_wrong = -1
					for item in homophonic:
						index = item[0]
						word = item[1]
						if word not in stops:
							v1 = self.get_vector([word])
							v2 = self.get_vector(get_context_words(index,words))
							sim = np.dot(v1,v2)
							if sim < cos_sim:
								cos_sim = sim
								index_wrong = index
								h_words = self.homophones[word]
							
				
							
			
			
		word = words[index_wrong]
		candidates = list(self.generateNonStopCandidates(word,stops))
		context = get_context_words(i,words)

		v1 = self.get_vector(context)
		v2 = self.get_all_vectors(word,candidates)
		word_vals = v2[0]
		word_vecs = v2[1]
		word_dist = v2[2]
			
		sims = np.dot(word_vecs,v1)
		sims = np.divide(sims,word_dist)
			
		idx = np.argsort(-sims)
			
		sorted_words = np.array(word_vals)[idx]
		sorted_sims = np.array(sims)[idx]
		
		if isHomophonic:
			sorted_words = h_words + sorted_words.tolist()
		
		return (sorted_words[:top_k],word,sorted_sims[:top_k])
			
			
		

# Arguments should be normalized vectors		
def get_cosine_sim(vec1,vec2):
	return 	np.dot(vec1,vec2.T)
	

def get_context_words(index,words):

	num_context = 2
	n = len(words)
	
	left = min(index,num_context)
	
	right = min(num_context,n-index-1)
	
	word_list = []
	
	for i in range(0,left):	
		word_list.append(words[index-1-i])
	
	for i in range(0,right):
		word_list.append(words[index+1+i])
		
	return word_list
	
def get_homophones_from_file(filename):
	fp = open(filename,'r')

	homophones = {}

	for line in fp:
		line = line.strip().split(',')
		for i in range(0,len(line)):
			for j in range(i+1,len(line)):
				if line[i] in homophones:
					if line[j] not in homophones[line[i]]:
						homophones[line[i]].append(line[j])
				else:
					homophones[line[i]] = [line[j]]
				
				if line[j] in homophones:
					if line[i] not in homophones[line[j]]:
						homophones[line[j]].append(line[i])
				else:
					homophones[line[j]] = [line[i]]
					
	return homophones
		
		
		
if __name__ == '__main__':

	if len(sys.argv) != 3:
		print "Usage: python sentences.py <infile> <outfile>"
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
		
		homophones = get_homophones_from_file('Data/Dictionaries/homophones.txt')
		
		# Build Checker model
		s_checker = SentenceChecker(word_set, unigrams, 1, vec_file, homophones, lamda=0.05)
		
		with open('Data/Models/sentence_model.pkl', 'wb') as fp:
			cPickle.dump(s_checker, fp)
			
	
	# Load Checker model
	
	with open('Data/Models/sentence_model.pkl', 'rb') as fp:
		s_checker = cPickle.load(fp)
	
	
	# For every sentence, for each word - generate candidates, take context words, form representations and then find scores for each candidate(upto 3, say)
		
	# Output results
	stops = set(stopwords.words('english'))
	with open(sys.argv[1]) as fin, open(sys.argv[2], 'w') as fout:
		for line in fin:
			sentence = line.strip().lower()
			
			words = sentence.strip().split()
			
			suggestions,wrong,scores = s_checker.correct(words,stops,alpha=1,top_k=3)
			if DEBUG:
				fout.write('\t'.join([wrong] + [word for word in suggestions]) + '\n')
				fout.write('\t'.join([str(score) for score in scores])+'\n')
			else:
				fout.write('\t'.join([wrong] + [word for word in suggestions]) + '\n')
	
	
		
		
