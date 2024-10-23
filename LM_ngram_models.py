"""
PART II:
In this assignment, you will train several language models and will evaluate them on a test
corpus. You can discuss in groups, but the homework is to be completed and submitted
individually. Two files are provided with this assignment:
1. train.txt
2. test.txt
Each file is a collection of texts, one sentence per line. train.txt contains about 100,000 sentences
from the NewsCrawl corpus. You will use this corpus to train the language models.
The test corpus test.txt is from the same domain and will be used to evaluate the language
models that you trained.
 """

import math     #used for log base 2 calculations

#file paths of tokenized txt files 
test_filename = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\test.txt"
train_filename = "C:\\Users\\sneeky\\Desktop\\Fall24Semester\\NLP\\train-Fall2024.txt"


#takes a tokenized file and returns a hashmap of the tokens lowercased, including sentence padding
def pad_and_hashmap(input_filename):

    token_hashmap = {}  #hashmap of tokens to return

    #attempt to open
    with open(input_filename, "r", encoding = "utf-8") as f:    

        for line in f:               #go line by line

            line_list = line.split()    #seperate by tokens into a list

            #go token by token lowercasing and adding </s> <s>
            for token in line_list:

                #if it's an end-of-sentence punctuation, add the padding
                if(token == "." or token == "?" or token == "!"):

                    #add to hashmap as appropriate
                    try:
                        token_hashmap[token] += 1
                    except:
                        token_hashmap[token] = 1

                    try:
                        token_hashmap["</s>"] +=1
                    except:
                        token_hashmap["</s>"] = 1

                    try:
                        token_hashmap["<s>"] +=1
                    except:
                        token_hashmap["<s>"] = 1

                #if the word is not an end of sentence, just lowercase
                else:

                    #add to hashmap as appropriate
                    try:
                        token_hashmap[token.lower()] += 1
                    except:
                        token_hashmap[token.lower()] = 1

    return token_hashmap    #return hashmap of tokens

#takes a tokenized file and hashmap. Returns file as list adding sentence padding and replacing singletons with <unk>
def list_with_unk(input_filename, token_amount, comparison_hashmap):

    corpus_list = [" "] *  (token_amount + 1)                         #create empty list to store corpus in chronological order
    corpus_list[0] = "<s>"                                            #start the list with a sentence begin token

    #attempt to open file
    with open(input_filename, "r", encoding = "utf-8") as f:

        token_counter = 1               #counter to keep track of tokens added to list

        #go line by line
        for line in f:

            line_list = line.split()    #seperate by tokens into a list

            #go token by token
            for token in line_list:
                token = token.lower()     #lowercase every word     

                token_occurrence = comparison_hashmap.get(token, 0)   #number of times the token appears in the training data

                #if the token occurs more than once add the token
                if(token_occurrence > 1):

                    #if there is an end-of-sentence punctuation, add the padding
                    if(token == "." or token == "?" or token == "!"):
                        corpus_list[token_counter] = token
                        corpus_list[token_counter+1] = "</s>"
                        corpus_list[token_counter+2] = "<s>"
                        token_counter+=3                    
                    
                    #if the token is not end of sentence
                    else:
                        corpus_list[token_counter] = token
                        token_counter +=1                                    

                #if the token occurs once or less
                else:
                    corpus_list[token_counter] = "<unk>"
                    token_counter+=1                     

    return corpus_list

#takes a list and creates a hashmap of each word occurence for use as  a unigram model
def unigramify(corpus_list2):
    
    unigram_hash = {}            #hashmap to store monogram model

    #create a hashmap of the count of words found in the corpus
    for i in range(0,len(corpus_list2)):
        try:
            unigram_hash[corpus_list2[i]] += 1
        except:
            unigram_hash[corpus_list2[i]] = 1
    
    return unigram_hash

#takes a list and creates a hashmap of each word pair occurence for use in bigram model
def bigramify(corpus_list2):
    
    bigram_hash = {}            #hashmap to store count for pairs of words

    #create a hashmap for count of every pair of words found in the corpus
    for i in range(1,len(corpus_list2)):
            try:
                bigram_hash[(corpus_list2[i], corpus_list2[i-1])] += 1
            except:
                bigram_hash[(corpus_list2[i], corpus_list2[i-1])] = 1
    
    return bigram_hash

#returns the probability of the word_to_find occuring under the unigram model of size corpus_word_count
def uni_prob(monogram_model, word_to_find, corpus_word_count):
    #P(word_to_find) = Count(word_to_find) / corpus_word_count (MLE)

    count_word = monogram_model.get(word_to_find, 0)    #number of count for the word treated as unk if 0
    prob = count_word / corpus_word_count                     #probability of word being selected

    #if probability is greater than 0, return the log probability
    if(prob > 0):
        return math.log(prob, 2)
    
    #if 0, return 0
    else:
        return 0

#returns the probability of the word pair occuring under the bigram model of size corpus_word_count, smoothing with add_n_smooth amount
def bi_prob(conditional_count, last_word_count, corpus_word_count, add_n_smooth):
    #P(next word | last_word) = (conditional_count + add_n_smooth) / (last_word_count + add_n_smooth * corpus_word_count) for all n \in \naturals_0

    conditional_count += add_n_smooth                   #count of word pair after smoothing
    last_word_count += add_n_smooth * corpus_word_count #count of previous word after smoothing

    #if denomenator is not 0 find probability
    conditional_prob = conditional_count / last_word_count

    #if positive return the log probability
    if(conditional_prob > 0):
        return math.log(conditional_prob, 2)
    
    #if 0 probability return 0
    else:
        return 0

"""
1.1 PRE-PROCESSING
 Prior to training, please complete the following pre-processing steps:

 1. Pad each sentence in the training and test corpora with start and end symbols (you can
 use <s> and </s>, respectively).

 2. Lowercase all words in the training and test corpora. Note that the data already has
 been tokenized (i.e. the punctuation has been split off words).
 
 3. Replace all words occurring in the training data once with the token <unk>. Every word
 in the test data not seen in training should be treated as <unk>.
"""

#hashmaps for respective data
train_token_hashmap = pad_and_hashmap(train_filename)     #create a hashmap of the train data
test_token_hashmap = pad_and_hashmap(test_filename)       #create a hashmap of the test data

#create post-processed training list
num_train_tokens = sum(train_token_hashmap.values())    #number of tokens in the training data
train_list = list_with_unk(train_filename, num_train_tokens, train_token_hashmap)
train_list[-1] = "</s>" #add final end of sentence token

"""
How many word types (unique words) are there in the training corpus? Please include
the end-of-sentence padding symbol </s> and the unknown token <unk>. 
Do not include the start of sentence padding symbol <s>.
"""

unique_train_tokens = set(train_list)                         #unique set of possible words
s_in_corpus = "<s>" in unique_train_tokens                    #true if <s> is corpus
wordtype_no_s = len(unique_train_tokens) - int(s_in_corpus)   #unique tokens excluding <s> when applicable

print("There are", wordtype_no_s, " word types in the training data")

"""
How many word tokens are there in the training corpus? Do not include the start of
sentence padding symbol <s>.
"""

s_occurrences = train_token_hashmap.get("<s>", 0)       #count of <s> in training corpus
token_count_no_s = num_train_tokens - s_occurrences     #count of non <s> tokens

print("There are", token_count_no_s, " word tokens in the training data")

""" 
What percentage of word tokens and word types in the test corpus did not occur in
training (before you mapped the unknown words to <unk> in training and test data)?
Please include the padding symbol </s> in your calculations. Do not include the start
of sentence padding symbol <s>.
 """

num_test_tokens = sum(test_token_hashmap.values())      #number of tokens in the test corpus
unique_test_tokens_no_unk = set(test_token_hashmap)     #set of unique tokens in the test corpus
test_only_words = 0                                     #unique words found only in test
num_test_only_words = 0                                 #total test_only_words occurences

#for every unique word in test, if it doesn't appear in the training corpus, increment counter and total count
for word in unique_test_tokens_no_unk:
    if(train_token_hashmap.get(word, 0) == 0):
        test_only_words += 1
        num_test_only_words += test_token_hashmap.get(word, 0)

#percent of wordtype not in training
test_only_wordtype_percent = 100 * test_only_words / len(unique_test_tokens_no_unk)

#percent of words not in training
test_only_count_percent = 100 * num_test_only_words / num_test_tokens

print(test_only_wordtype_percent, "percent of word types did not appear in training")
print(test_only_count_percent, "percent of word tokens did not appear in training")

"""
Now replace singletons in the training data with <unk> symbol and map words (in the
 test corpus) not observed in training to <unk>. What percentage of bigrams (bigram
 types and bigram tokens) in the test corpus did not occur in training (treat <unk> as a
 regular token that has been observed). Please include the padding symbol </s> in your
 calculations. Do not include the start of sentence padding symbol <s>.
 """

test_list = list_with_unk(test_filename, num_test_tokens, train_token_hashmap)   #create post-processed list
test_list.pop()
unique_test_tokens = set(test_list)                             #unique tokens in post-processed list

train_unigram = unigramify(train_list)                         #create unigram model hashmap training data
train_bigram = bigramify(train_list)                            #create bigram model hashmap for training data

test_bigram = bigramify(test_list)                              #create bigram model for testing data

num_new_bigram_type = 0                                         #unique number of new word pairs
num_new_bigram_count = 0                                        #count of new word pairs 

#test all word pairs in test model to see if they're new
for bigram_token in test_bigram:

    #exclude pairs involving <s> for this calculation
    if(bigram_token[0] == "<s>" or bigram_token[1] == "<s>"):
        pass

    #otherwise test to see if its in the training model
    else:
        #if the pair is new, increment unique type and add its respective value to the new count
        if(bigram_token not in train_bigram):
            num_new_bigram_type += 1
            num_new_bigram_count += test_bigram.get(bigram_token)

#percent of test corpus that is new (counter/total)
num_new_bigram_type_percent = 100 * num_new_bigram_type / len(test_bigram)
num_new_bigram_count_percent = 100 * num_new_bigram_count / sum(test_bigram.values())

print(num_new_bigram_type_percent, "percent new bigram types")
print(num_new_bigram_count_percent, "percent new bigram tokens")

""" 
Compute the log probability of the following sentence under the three models (ignore
capitalization and pad each sentence as described above). Please list all of the parameters
required to compute the probabilities and show the complete calculation. Which
of the parameters have zero values under each model? Use log base 2 in your calculations.
Map words not observed in the training corpus to the <unk> token.

I look forward to hearing your reply.
"""

#add padding to the sentence and split into a list
sentence_string = "I look forward to hearing your reply ."
sentence_string = "<s> " + sentence_string + " </s>"
sentence_string = sentence_string.split() 

#lowercase words replacing new words with <unk>
for i in range(0, len(sentence_string)):
    sentence_string[i] = sentence_string[i].lower()
    if(train_token_hashmap.get(sentence_string[i], 0) == 0):
        sentence_string[i] = "<unk>"

uni_prob_sentence = 0     #monogram probability
bi_prob_sentence = 0      #bigram probability
bi_add1_prob_sentence = 0 #bigram add one probability

count_unk = train_unigram.get("<unk>")     #counter of <unk> for new test-only words
unseen_word_pair = 0                        #dummy variable for if a new bigram token was seen

#sum the log probabilities for each word
for word in sentence_string:

    word_prob = uni_prob(train_unigram, word, num_train_tokens)
    print("Prob under unigram for (", word, ") is", word_prob )
    uni_prob_sentence += word_prob

#sum the log probabilities for each word pair
for i in range(1, len(sentence_string)):
    next_word = sentence_string[i]
    last_word = sentence_string[i-1]

    word_tuple_count = train_bigram.get((next_word, last_word), 0)          #word pair count
    count_prev_word = train_unigram.get(last_word, 0)                 #previous word count

    word_pair_prob = bi_prob(word_tuple_count, count_prev_word, len(unique_train_tokens), 0)
    add1_word_pair_prob = bi_prob(word_tuple_count, count_prev_word, len(unique_train_tokens), 1)

    bi_prob_sentence += word_pair_prob              #sum for bigram model
    bi_add1_prob_sentence += add1_word_pair_prob    #sum for bigram add-one model

    if(word_tuple_count == 0):
        unseen_word_pair = 1
        word_pair_prob = "undefined"

    print("Prob under bigram for (", next_word, "|", last_word, ") is", word_pair_prob)
    print("Prob under bigram_add1 for (", next_word, "|", last_word, ") is", add1_word_pair_prob)

if(unseen_word_pair == 1):
    bi_prob_sentence = "undefined"

print("total log probability for unigram model is ", uni_prob_sentence)
print("total log probability for bigram model is ", bi_prob_sentence)
print("total log probability for bigram add-one model is ", bi_add1_prob_sentence)

""" 
Compute the perplexity of the sentence above under each of the models.
"""

#use perplexity formula to find respective perplexity
mono_perplex_sentence = math.pow(2, -uni_prob_sentence/len(sentence_string))

try:
    bi_perplex_sentence = math.pow(2, -bi_prob_sentence/len(sentence_string))
except:
    bi_perplex_sentence = "undefined"

bi_add1_perplex_sentence = math.pow(2, -bi_add1_prob_sentence/len(sentence_string))

print("the perplexity of the sentence under the unigram model is", mono_perplex_sentence)
print("the perplexity of the sentence under the bigram model is", bi_perplex_sentence)
print("the perplexity of the sentence under the bigram add-one model is", bi_add1_perplex_sentence)

"""
Compute the perplexity of the entire test corpus under each of the models. Discuss the
differences in the results you obtained.
"""

uni_corpus_prob = 0.0
bi_prob_corpus = 0.0
bi_add1_prob_corpus = 0.0

unseen_word_pair = 0
unseen_combo = 0

#sum of log probabilities under monogram model
for word in test_list:
    uni_corpus_prob += uni_prob(train_unigram, word, num_train_tokens)

#sum of log probabilities under bigram models
for i in range(1, len(test_list)):
    word_tuple_count = train_bigram.get((test_list[i], test_list[i-1]), 0)  #count of word pair
    count_prev_word = train_unigram.get(test_list[i-1], count_unk)             #count of previous word


    bi_prob_corpus += bi_prob(word_tuple_count, count_prev_word, num_train_tokens, 0)       #bigram probability
    bi_add1_prob_corpus += bi_prob(word_tuple_count, count_prev_word, num_train_tokens, 1)  #bigram add one probability

    #if a new bigram token was seen, set the dummy to 1
    if(word_tuple_count == 0):
        unseen_word_pair = 1

#calculate perplexities
uni_perplex_corpus = math.pow(2, -uni_corpus_prob / num_test_tokens)
bi_perplex_corpus = math.pow(2, -bi_prob_corpus / num_test_tokens)
bi_add1_perplex_corpus = math.pow(2, -bi_add1_prob_corpus / num_test_tokens)

#if a new bigram token was seen, set the bigram perplexity to 0
if(unseen_word_pair == 1):
    bi_perplex_corpus = "undefined"

print("the perplexity of the corpus under the unigram model is", uni_perplex_corpus)
print("the perplexity of the corpus under the bigram model is",bi_perplex_corpus)
print("the perplexity of the corpus under the bigram add-one model is",bi_add1_perplex_corpus)
