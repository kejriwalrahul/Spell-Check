"""
	Program to estimate costs of different edits for Damerau-Levenshtein Edit Distance

	Rahul Kejriwal, CS14B023
"""

import numpy as np

WEIGH_FACTOR  = 10
WEIGH_FACTOR2 = 3

# Read spelling error pairs with 1 edit dist btwn them
with open('Data/Errors/1edit.dat') as fp:
	pairs = [line.strip().split(',') for line in fp]

# pairs = pairs[:1000]

# To Store frequency of edits seen in incorrect spellings
insert_counts = {}
delete_counts = {}
substitute_counts = {}
transpose_counts = {}

# Locate error and increment corresponding count
for w,c in pairs:
	# insertion
	if len(w) > len(c):
		flag = False
		for i,ch in enumerate(c):
			if w[i]!=ch:
				if w[i] not in insert_counts:
					insert_counts[w[i]] = 0	
				insert_counts[w[i]] += 1
				flag = True
				break
		if not flag:
			if w[-1] not in insert_counts:
				insert_counts[w[-1]] = 0	
			insert_counts[w[-1]] += 1
	# deletion
	elif len(w) < len(c):
		flag = False
		for i,ch in enumerate(w):
			if c[i]!=ch:
				if c[i] not in delete_counts:
					delete_counts[c[i]] = 0	
				delete_counts[c[i]] += 1
				flag = True
				break
		if not flag:
			if c[-1] not in delete_counts:
				delete_counts[c[-1]] = 0	
			delete_counts[c[-1]] += 1
	# transposition or substitution
	else:
		diff=[]
		for i, ch in enumerate(c):
			if w[i] != ch:
				diff.append((w[i],ch))

		# substitution
		if len(diff) == 1:
			if diff[0] not in substitute_counts:
				substitute_counts[diff[0]] = 0
			substitute_counts[diff[0]] += 1
		# Transposition
		elif len(diff) == 2:
			if (diff[0][0]+diff[1][0],diff[0][1]+diff[1][1]) not in transpose_counts:
				transpose_counts[(diff[0][0]+diff[1][0],diff[0][1]+diff[1][1])] = 0
			transpose_counts[(diff[0][0]+diff[1][0],diff[0][1]+diff[1][1])] += 1
		else:
			raise NotImplementedError

# Update insert counts after smoothing (Add-1)
ins_costs = np.ones((128,)) * WEIGH_FACTOR
for ch in insert_counts:
	ins_costs[ord(ch)] += insert_counts[ch]

# Update delete counts after smoothing (Add-1)
del_costs = np.ones((128,)) * WEIGH_FACTOR
for ch in delete_counts:
	del_costs[ord(ch)] += delete_counts[ch]

# Update substitution counts after smoothing (Add-1)
sub_costs = np.ones((128,128)) * WEIGH_FACTOR
for w_ch, c_ch in substitute_counts:
	sub_costs[ord(w_ch), ord(c_ch)] += substitute_counts[(w_ch, c_ch)] 

# Update transposition counts after smoothing (Add-1)
trans_costs = np.ones((128,128)) * WEIGH_FACTOR2
for w_cs, c_cs in transpose_counts:
	trans_costs[ord(w_cs[0]), ord(w_cs[1])] += transpose_counts[(w_cs, c_cs)]

total = np.sum(ins_costs) + np.sum(del_costs) + np.sum(sub_costs) + np.sum(trans_costs)

ins_costs = 6.9 - np.log(ins_costs) 
del_costs = 6.9 -np.log(del_costs) 
sub_costs = 6.9 - np.log(sub_costs) 
trans_costs = 6.9 - np.log(trans_costs) 

print ins_costs
print del_costs
print sub_costs
print trans_costs

# Save costs to file
with open('Data/costs.npz', 'w') as fp:
	np.savez(fp, ins_costs=ins_costs, del_costs=del_costs, sub_costs=sub_costs, trans_costs=trans_costs)