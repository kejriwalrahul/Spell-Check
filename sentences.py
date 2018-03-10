from correct import *

class SentenceChecker(SpellChecker):
	
	def __init__(self,word_set, unigrams, k, edit_counts, vector_file, lamda=1, alphabet='abcdefghijklmnopqrstuvwxyz'):
		SpellChecker.__init__(self,word_set, unigrams, k, edit_counts, lamda)
		
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
			
		vec_result = vec_result/float(count) 
		
		vec_norm = np.zeros(vec_result.shape)
		d = (np.sum(vec_result ** 2,) ** (0.5))
		vec_norm = (vec_result.T / d).T
		
		return vec_norm
		
	def get_all_vectors(self,wrong,words):
		
		word_vals = [term for term in words if term in self.vocab]
		word_vec = np.array([self.W_norm[self.vocab[term], :]/((dam_lev(wrong, term, insert_costs=self.insert_costs, substitute_costs=self.substitute_costs, delete_costs=self.delete_costs, transpose_costs=self.transpose_costs))) for term in words if term in self.vocab])
		
		return (word_vals,word_vec)
		
	
	# for each word - generate candidates, take context words, form representations and then find scores for each candidate(upto 3, say)
	def correct(self, words, top_k=3):
		for i,word in enumerate(words):
			candidates = list(self.generateCandidates(word))
			context = get_context_words(i,words)
			#print candidates
			#print context
			#print "###"
			v1 = self.get_vector(context)
			v2 = self.get_all_vectors(word,candidates)
			word_vals = v2[0]
			word_vecs = v2[1]
			
			sims = np.dot(word_vecs,v1)
			
			idx = np.argsort(-sims)
			
			sorted_words = np.array(word_vals)[idx]
			sorted_sims = np.array(sims)[idx]
			
			print sorted_words[:20]
			print sorted_sims[:20]
			print "###"
			
			
		

# Arguments should be normalized vectors		
def get_cosine_sim(vec1,vec2):
	return 	np.dot(vec1,vec2.T)
	

def get_context_words(index,words):
	n = len(words)
	
	left = min(index,4)
	
	right = min(4,n-index-1)
	
	word_list = []
	
	for i in range(0,left):	
		word_list.append(words[index-1-i])
	
	for i in range(0,right):
		word_list.append(words[index+1+i])
		
	return word_list
		
		
		
if __name__ == '__main__':
	
	FRESH = False
	
	if FRESH:

		# Read dictionaries for candidate generation
		word_set = read_csv_dict('Data/Dictionaries/dictionary.csv')
		word_set = word_set.union(read_list_dict('Data/Dictionaries/word.list'))

		# Read unigram counts for prior/LM model
		unigrams = read_unigram_probs('Data/count_1w.txt') 

		# Read edit counts for likelihood/channel model
		edit_counts = read_edit_counts('Data/count_1edit.txt')
		
		vec_file = 'Data/Vectors/glove.6B.50d.txt'
		
		# Build Checker model
		s_checker = SentenceChecker(word_set, unigrams, 1, edit_counts, vec_file, lamda=0.05)
		
		with open('sentence_model.pkl', 'wb') as fp:
			cPickle.dump(s_checker, fp)
			
	
	# Load Checker model
	
	with open('sentence_model.pkl', 'rb') as fp:
		s_checker = cPickle.load(fp)
	
	
	# For every sentence, for each word - generate candidates, take context words, form representations and then find scores for each candidate(upto 3, say)
	while True:
		sentence = raw_input("Enter a sentence: ")
		if sentence == 'EXIT':
			break
		words = sentence.strip().split()
	
		s_checker.correct(words)
	
	
		
		
