import io, sys
import numpy as np
from heapq import *

def load_vectors(filename):
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(list(map(float, tokens[1:])))
    return data

## This function computes the cosine similarity between vectors u and v

def cosine(u, v):
    ## FILL CODE
    return u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v))

## This function returns the word corresponding to 
## nearest neighbor vector of x
## The list exclude_words can be used to exclude some
## words from the nearest neighbors search

def nearest_neighbor(x, word_vectors, exclude_words=[]):
    best_score = -1.0
    best_word = ''

    ## FILL CODE
    for word in word_vectors:
        if word not in exclude_words:
            score = cosine(x, word_vectors[word])
            if score > best_score:
                best_score = score
                best_word = word

    return best_word

## This function return the words corresponding to the
## K nearest neighbors of vector x.
## You can use the functions heappush and heappop.

def knn(x, vectors, k):
    heap = []
    ## FILL CODE
    for word in vectors:
        if (x != vectors[word]).any(): # To exclude the word itself
            if len(heap) < k: # Check if the heap length is still less than k, then add the next word
                heappush(heap, (cosine(x, vectors[word]),word))
            else:
                heapify(heap) # We send the word with smallest similarity at index 0
                if heap[0][0] < cosine(x, vectors[word]): # If the new word is more similar to the target as compared to the word t index 0, then replace
                    heapreplace(heap, (cosine(x, vectors[word]), word))  
    return [heappop(heap) for i in range(len(heap))][::-1] #return the heap in descending order


## This function return the word d, such that a:b and c:d
## verifies the same relation

def analogy(a, b, c, word_vectors):
    ## FILL CODE
    a, b, c = a.lower(), b.lower(), c.lower()
    target = 0
    best = -np.inf
    best_word = None
    for word in word_vectors:
        if word not in [a, b, c]:
            if (word_vectors[b]-word_vectors[a]+word_vectors[c]).dot(word_vectors[word]) > best:
                best = (word_vectors[b]-word_vectors[a]+word_vectors[c]).dot(word_vectors[word])
                best_word = word
    return best_word

## Compute the association strength between:
##   - a word w
##   - two sets of attributes A and B

def association_strength(w, A, B, vectors):
    strength = (1./len(A))*np.sum([cosine(vectors[a], vectors[w]) for a in A])-(1./len(B))*np.sum([cosine(vectors[b], vectors[w]) for b in B])
    return strength

## Perform the word embedding association test between:
##   - two sets of words X and Y
##   - two sets of attributes A and B

def weat(X, Y, A, B, vectors):
    ## FILL CODE
    score = np.sum([association_strength(x, A, B, vectors) for x in X])-np.sum([association_strength(y, A, B, vectors) for y in Y])
    return score

######## MAIN ########

print('')
print(' ** Word vectors ** ')
print('')

#word_vectors = load_vectors(sys.argv[1])
word_vectors=load_vectors('wiki.en.vec')

print('similarity(apple, apples) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['apples']))
print('similarity(apple, banana) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['banana']))
print('similarity(apple, tiger) = %.3f' %
      cosine(word_vectors['apple'], word_vectors['tiger']))

print('')
print('The nearest neighbor of cat is: ' +
      nearest_neighbor(word_vectors['cat'], word_vectors))

knn_cat = knn(word_vectors['cat'], word_vectors, 5)
print('')
print('cat')
print('--------------')
for score, word in knn(word_vectors['cat'], word_vectors, 5):
    print (word + '\t%.3f' % score)

print('')
print('france - paris + rome = ' + analogy('paris', 'france', 'rome', word_vectors))
print('')
print('king - man + woman = ' + analogy('man', 'king', 'woman', word_vectors))

## A word about biases in word vectors:

print('')
print('similarity(genius, man) = %.3f' %
      cosine(word_vectors['man'], word_vectors['genius']))
print('similarity(genius, woman) = %.3f' %
      cosine(word_vectors['woman'], word_vectors['genius']))

## Replicate one of the experiments from:
##
## Semantics derived automatically from language corpora contain human-like biases
## Caliskan, Bryson, Narayanan (2017)

career = ['executive', 'management', 'professional', 'corporation', 
          'salary', 'office', 'business', 'career']
family = ['home', 'parents', 'children', 'family',
          'cousins', 'marriage', 'wedding', 'relatives']
male = ['john', 'paul', 'mike', 'kevin', 'steve', 'greg', 'jeff', 'bill']
female = ['amy', 'joan', 'lisa', 'sarah', 'diana', 'kate', 'ann', 'donna']

print('')
print('Word embedding association test: %.3f' %
      weat(career, family, male, female, word_vectors))
print('')