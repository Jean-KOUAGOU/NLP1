import io, sys, math, re
from collections import defaultdict
import numpy as np

# GOAL: build a stupid backoff ngram model

def load_data(filename):
    fin = io.open(filename, 'r', encoding='utf-8')
    data = []
    vocab = defaultdict(lambda:0)
    for line in fin:
        sentence = line.split()
        data.append(sentence)
        for word in sentence:
            vocab[word] += 1
    return data, vocab

def remove_rare_words(data, vocab, mincount=0):
    ## FILL CODE
    # replace words in data that are not in the vocab 
    # or have a count that is below mincount
    data_with_unk = data[:]
    for i in range(len(data_with_unk)):
        for j in range(len(data_with_unk[i])):
            if vocab[data_with_unk[i][j]] < mincount:
                data_with_unk[i][j] = '<unk>'
    return data_with_unk


def build_ngram(data, n):
    total_number_words = 0
    counts = defaultdict(lambda: defaultdict(lambda: 0.0))

    for sentence in data:
        sentence = tuple(sentence)
        ## FILL CODE
        # dict can be indexed by tuples
        # store in the same dict all the ngrams
        # by using the context as a key and the word as a value
        for gram_size in range(n):
            for idx in range(len(sentence)):
                total_number_words += 1.
                if gram_size+idx < len(sentence):
                    counts[sentence[idx:gram_size+idx]][sentence[idx+gram_size]] += 1.
    total_number_words /= n #This quantity was n times the actual one

    prob  = defaultdict(lambda: defaultdict(lambda: 0.0))
    ## FILL CODE
    # Build the probabilities from the counts
    # Be careful with how you normalize!
    for context in counts:
        for word in counts[context]:
            prob[context][word] = counts[context][word]/sum(counts[context].values())

    return prob

def get_prob(model, context, w):
    ## FILL CODE
    # code a recursive function over 
    # smaller and smaller context
    # to compute the backoff model
    return model[context][w] if model[context][w] != 0.0 else 0.4*get_prob(model, context[1:], w)

    # Bonus: You can also code an interpolation model this way
def get_prob_bonus(model, context, w):
    n = max([len(key) for key in model])+1
    lambda_s = 1./n
    s = len(context)
    probs = 0.0
    for i in range(n):
        probs += lambda_s*get_prob(model, context[s-1-i:], w)
    return probs
    

def perplexity(model, data, n):
    ## FILL CODE
    # Same as bigram.py
    perp = 0.0
    for sentence in data:
        sentence = tuple(sentence)
        probs = 0.0
        for idx in range(1,len(sentence)):
            probs += (-1.0/len(sentence))*np.log(get_prob(model, sentence[max(0,idx-n+1):idx], sentence[idx]))
        perp += probs/len(data)
    return np.exp(perp)

def get_proba_distrib(model, context):
    ## FILL CODE
    # code a recursive function over context
    # to find the longest available ngram 
    return context if context in model else get_proba_distrib(model, context[1:])

def generate(model):
    sentence = ["<s>"]
    ## FILL CODE
    # generate a sentence. A sentence starts with a <s> and ends with a </s>
    # Possiblly a use function is:
    #   np.random.choice(x, 1, p = y)
    # where x is a list of things to sample from
    # and y is a list of probability (of the same length as x)
    sentence = ['<s>']
    while sentence[-1] != '</s>':
        x = list(model[get_proba_distrib(model, tuple(sentence))].keys())
        proba = list(model[get_proba_distrib(model, tuple(sentence))].values())
        #Now we sample a word from x according to proba
        sentence.append(np.random.choice(x, 1, p = proba)[0])
    return sentence

###### MAIN #######

n = 2

print("load training set")
train_data, vocab = load_data("train.txt")

## FILL CODE
# Same as bigram.py
train_data = remove_rare_words(train_data, vocab, 5)

print("build ngram model with n = ", n)
model = build_ngram(train_data, n)

print("load validation set")
valid_data, _ = load_data("valid.txt")
## FILL CODE
# Same as bigram.py
valid_data = remove_rare_words(valid_data, vocab, 5)

print("The perplexity is", perplexity(model, valid_data, n))

print("Generated sentence: ",generate(model))

