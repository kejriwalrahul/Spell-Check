fp = open('sentences_val.txt','r')
fpw = open('validation_sentences.txt','w')

for line in fp:
	line = line.strip().split('.')
	fpw.write(line[0]+'\n')
	
fp.close()
fpw.close()
	
