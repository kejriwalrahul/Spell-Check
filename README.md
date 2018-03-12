# Spell-Check
Spelling Correction using Noisy Channel Models

## Requirements
1. Fuzzy==1.2.2
2. weighted-levenshtein==0.1
3. cPickle
4. tqdm
5. numpy

## Usage:

1. To run word-level spell-correction, run `python correct.py <input_file_path> <output_file_path>`

## Sentence correction
Store the GloVe word vectors file (can be found at https://nlp.stanford.edu/projects/glove/) in the `Data/Vectors/` folder.

## Things to do:
1. Sometimes suggestions are same as the wrong word, also avoid repeated suggestions
2. Need a better dictionary
3. Need a better way to identify the wrong word (rather than just context vector)
4. Tune a few weights of the model
5. Try using language model probabilities (trigram/quadrigram) to find word to be corrected for phrases/sentences rather than ad-hoc schemes/manual rules