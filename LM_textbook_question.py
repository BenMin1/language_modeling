"""
PART I:

We are given the following corpus, modified from the one in the chapter:
<s> I am Sam </s>
<s> Sam I am </s>
<s> I am Sam </s>
<s> I do not like green eggs and Sam </s>

Using a bigram language model with add-one smoothing, what is P(Sam | am)? 
Include <s> and </s> in your counts just like any other token.
"""

corpus = "<s> I am Sam </s> <s> Sam I am </s> <s> I am Sam </s> <s> I do not like green eggs and Sam </s>"

tokens = corpus.split()     #tokenize corpus into list
unique_tokens = set(tokens) #unique set of possible words

bigram_hash = {}            #hashmap to store count for pairs of words

#create a hashmap for every pair of words found in the corpus
for i in range(1,len(tokens)):
    try:
        bigram_hash[(tokens[i], tokens[i-1])] += 1
    except:
        bigram_hash[(tokens[i], tokens[i-1])] = 1
    
conditional_hash = {}       #hashmap of next word counts, given previous word and add one smoothing

#fill the conditional hashmap
for word in unique_tokens:
    try:        #if the word pair exists, add one 
        conditional_hash[word] = bigram_hash[(word, "am")] + 1
    except:     #if the word pair doesn't exist, set it to 1
        conditional_hash[word] = 1

print("There is a", conditional_hash["Sam"], "out of", sum(conditional_hash.values()), "chance \"Sam\" will be next")

#There is a 3/14 chance that "Sam" will be next
