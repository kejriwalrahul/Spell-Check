# Spell-Check
Spelling Correction using Noisy Channel Models

## Requirements
1. Fuzzy==1.2.2
2. weighted-levenshtein==0.1
3. cPickle
4. tqdm
5. numpy
6. nltk (for stopwords removal)

## Usage:
1. To run word-level spell-correction, run `python correct.py <input_file_path> <output_file_path>`
2. To run phrase-level spell-correction, run `python phrases.py <input_file_path> <output_file_path>`
2. To run sentence-level spell-correction, run `python sentences.py <input_file_path> <output_file_path>`
