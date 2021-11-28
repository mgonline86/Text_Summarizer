# coding=utf8

import os
import math
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
import pyarabic.araby as araby
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import networkx as nx
from bidi.algorithm import get_display
import arabic_reshaper
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph


static_path = 'static'
img_path = 'img'
file_top_2_bottom_map = 'top_2_bottom_map.png'
file_central_map = 'central_map.png'

path_top_2_bottom_map = os.path.join(static_path, img_path, file_top_2_bottom_map)
path_central_map = os.path.join(static_path, img_path, file_central_map)


global_stemmer = SnowballStemmer("arabic", ignore_stopwords=True)

# Find Weighted Frequency of Occurrence
stopwords = nltk.corpus.stopwords.words(['arabic','english'])

#--- Add extra arabic stop words ---#
#-----------------------------------#

# Adding kann_we_akhawateha
kann_we_akhawateha = ['كان', 'برح', 'دام', 'أصبح' ,'انفك' ,'ليس', 'أضحى', 'زال', 'ظل' ,'أمسى' ,'فتئ' ,'بات', 'ظل', 'صار']

# Adding anna_we_akhawateha
anna_we_akhawateha = ['إِن', 'أَن', 'كأن', 'لكن', 'ليت', 'لعل', 'لا']

# Adding zanna_we_akhawateha
zanna_we_akhawateha = ['علم', 'رأى', 'وجد', 'درى', 'ألفى', 'ظن', 'خال', 'حسب', 'زعم', 'عد', 'اعتبر', 'هب']

# Adding my_custom_list
my_custom_list = ['ربما', 'تكون', 'كون', 'فقد', 'وقد', 'والتي', 'قبل', 'لذا', 'وذلك', 'تم', 'فتم', 'يتم', 'فهم', 'فهو', 'فهي', 'ضمن', 'خلال']

stopwords += kann_we_akhawateha + anna_we_akhawateha + zanna_we_akhawateha + my_custom_list

# Replicate stopwords containing 'ي' and replacing the 'ي' with 'ى'
ye_stopwords = list(filter(lambda x:'ي' in x, stopwords))
ye_stopwords_replaced = list(map(lambda x:x.replace('ي', 'ى'), ye_stopwords))

# Replicate stopwords containing 'ى' and replacing the 'ى' with 'ي'
ya_stopwords = list(filter(lambda x:'ى' in x, stopwords))
ya_stopwords_replaced = list(map(lambda x:x.replace('ى', 'ي'), ya_stopwords))

# Replicate stopwords containing 'ا' and replacing the 'ا' with 'أ'
a_stopwords = list(filter(lambda x:'ا' in x, stopwords))
a_stopwords_replaced = list(map(lambda x:x.replace('ا', 'أ'), a_stopwords))

# Replicate stopwords containing 'أ' and replacing the 'أ' with 'ا'
aa_stopwords = list(filter(lambda x:'أ' in x, stopwords))
aa_stopwords_replaced = list(map(lambda x:x.replace('أ', 'ا'), aa_stopwords))


stopwords += ye_stopwords_replaced + ya_stopwords_replaced + a_stopwords_replaced + aa_stopwords_replaced

def delete_multiple_element(list_object, indices, final_sentence_list, remove_char_list):
    for i,e in enumerate(final_sentence_list):
        if e in remove_char_list:
            del final_sentence_list[i]

    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def get_sent_voc(sent_words_list, word_frequencies_stem):
    dic = {}
    for word in sent_words_list:
        if (stemmer.stem(word) not in stopwords) and (word not in stopwords) and (len(stemmer.stem(word)) > 0):
            dic[stemmer.stem(word)] = word_frequencies_stem[stemmer.stem(word)]
    return dic

class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """
 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
 
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
 
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word


stemmer = StemmingHelper()


def word2vec(text):
    text_input = araby.strip_diacritics(text)

    # Removing special characters and digits
    formatted_text = re.sub('[^\u0600-\u06FFa-zA-Z0-9]', ' ', text_input)
    formatted_text = re.sub(r'\s+', ' ', formatted_text)
    formatted_text = formatted_text.replace("،", " ")
    formatted_text = formatted_text.replace("\u061B", " ")

    # Converting Text To Sentences
    sentence_list = re.split(r'([.,،])', text_input)

    final_sentence_list = []
    for sent in sentence_list:
        sent = sent.replace("،", " ")
        sent = sent.replace("؟", " ")
        sent = sent.replace("\u061B", " ")
        final_sentence_list.append(re.sub(r"[^\u0600-\u06FFa-zA-Z0-9]+", ' ', sent))

    remove_char_list = ['', '.', ',', '،', ' ', '؟']
    remove_indices_list = []

    delete_multiple_element(final_sentence_list, remove_indices_list, final_sentence_list, remove_char_list)

    final_sentence_list = list(filter(None, final_sentence_list))

    sent2Vec = [nltk.word_tokenize(sent) for sent in final_sentence_list]


    sentence_terms = []

    word_frequencies_stem = {}
    for word in nltk.word_tokenize(formatted_text):
        if (stemmer.stem(word) not in stopwords) and (word not in stopwords) and (len(stemmer.stem(word)) > 0):
            if stemmer.stem(word) not in word_frequencies_stem.keys():
                word_frequencies_stem[stemmer.stem(word)] = 1
            else:
                word_frequencies_stem[stemmer.stem(word)] += 1



    for i, sent in enumerate(final_sentence_list):
        sentence_terms.append(get_sent_voc(sent2Vec[i],word_frequencies_stem))

    size = int(math.sqrt(len(sent2Vec)))
    min_count = 2
    window = 4

    model = Word2Vec(sent2Vec, min_count=min_count, vector_size=size, window=window)


    for i, s in enumerate(sentence_terms):
        for w in s.copy().keys():
            if w not in model.wv.key_to_index.keys():
                sentence_terms[i].pop(w, None)


    for w in word_frequencies_stem.copy():
        if w not in model.wv.key_to_index.keys():
            word_frequencies_stem.pop(w, None)

    sentence_terms = list(filter(lambda dic: len(dic.keys()) != 0, sentence_terms))

    return word_frequencies_stem, sentence_terms, model

def _get_param_matrices(vocabulary, sentence_terms):
    """
    Returns
    =======
    1. Top 300(or lesser, if vocab is short) most frequent terms(list)
    2. co-occurence matrix wrt the most frequent terms(dict)
    3. Dict containing Pg of most-frequent terms(dict)
    4. nw(no of terms affected) of each term(dict)
    """

    #Figure out top n terms with respect to mere occurences
    n = min(300, len(vocabulary))
    topterms = list(vocabulary.keys())
    topterms.sort(key = lambda x: vocabulary[x], reverse = True)
    topterms = topterms[:n]

    #nw maps term to the number of terms it 'affects'
    #(sum of number of terms in all sentences it
    #appears in)
    nw = {}
    #Co-occurence values are wrt top terms only
    co_occur = {}
    #Initially, co-occurence matrix is empty
    for x in vocabulary:
        co_occur[x] = [0 for i in range(len(topterms))]

    #Iterate over list of all sentences' vocabulary dictionaries
    #Build the co-occurence matrix
    for sentence in sentence_terms:
        total_terms = sum(list(sentence.values()))
        #This list contains the indices of all terms from topterms,
        #that are present in this sentence
        top_indices = []
        #Populate top_indices
        top_indices = [topterms.index(x) for x in sentence
                       if x in topterms]
        #Update nw dict, and co-occurence matrix
        for term in sentence:
            nw[term] = nw.get(term, 0) + total_terms
            for index in top_indices:
                co_occur[term][index] += (sentence[term] *
                                          sentence[topterms[index]])

    #Pg is just nw[term]/total vocabulary of text
    Pg = {}
    N = sum(list(vocabulary.values()))
    for x in topterms:
        Pg[x] = float(nw[x])/N

    return topterms, co_occur, Pg, nw



def get_top_n_terms(vocabulary, sentence_terms, n):
    """
    Returns the top 'n' terms from a block of text, in the form of a list,
    from most important to least.

    'vocabulary' should be a dict mapping each term to the number
    of its occurences in the entire text.
    'sentence_terms' should be an iterable of dicts, each denoting the
    vocabulary of the corresponding sentence.
    """

    #First compute the matrices
    topterms, co_occur, Pg, nw = _get_param_matrices(vocabulary,
                                                     sentence_terms)

    #This dict will map each term to its weightage with respect to the
    #document
    result = {}

    N = sum(list(vocabulary.values()))
    #Iterates over all terms in vocabulary
    for term in co_occur:
        term = str(term)
        org_term = str(term)
        for x in Pg:
            #expected_cooccur is the expected cooccurence of term with this
            #term, based on nw value of this and Pg value of the other
            expected_cooccur = nw[term] * Pg[x]
            #Result measures the difference(in no of terms) of expected
            #cooccurence and  actual cooccurence
            result[org_term] = ((co_occur[term][topterms.index(x)] -
                                 expected_cooccur)**2/ float(expected_cooccur))

    terms = list(result.keys())
    terms.sort(key=lambda x: result[x],
               reverse=True)

    return terms[:n]


######## Generate the actual Mind-Map ########


# Arabic text preprocessing 
def reshaped_arabic_text(txt):
    reshaped_text = arabic_reshaper.reshape(txt)
    return get_display(reshaped_text)

def del_ara_link_char(word):
    if len(word) > 1:
        if word[0] == 'و':
            if word[0] != stemmer.stem(word)[0]:
                return word[1:]
            elif stemmer.stem(word)[0] == 'و' and word[1] != stemmer.stem(word)[1]:
                return word[1:]
            else:
                return word
        elif word[0] == 'ل' and word[1] != 'ل':
            if word[0] != stemmer.stem(word)[0]:
                return word[1:]
            elif stemmer.stem(word)[0] == 'ل' and word[1] != stemmer.stem(word)[1]:
                return word[1:]
            else:
                return word
        elif word[0] == 'ل' and word[1] == 'ل':
            if word[0] != stemmer.stem(word)[0]:
                return word[2:]
            else:
                return word
        else:
            return word
    else:
        return word

def build_mind_map(model, root, nodes, alpha=0.2):
    """
    Returns the Mind-Map in the form of a NetworkX Graph instance.
 
    'model' should be an instance of gensim.models.Word2Vec
    'nodes' should be a list of terms, included in the vocabulary of
    'model'.
    'root' should be the node that is to be used as the root of the Mind
    Map graph.
    'stemmer' should be an instance of StemmingHelper.
    """
 
    #This will be the Mind-Map
    g = nx.DiGraph()

    #Ensure that the every node is in the vocabulary of the Word2Vec
    #model, and that the root itself is included in the given nodes
    for node in nodes:
        if node not in model.wv.key_to_index.keys():
            raise ValueError(node + " not in model's vocabulary")
    if root not in nodes:
        raise ValueError("root not in nodes")
 
    ##Containers for algorithm run
    #Initially, all nodes are unvisited
    unvisited_nodes = set(nodes)
    #Initially, no nodes are visited
    visited_nodes = set([])
    #The following will map visited node to its contextual vector
    visited_node_vectors = {}
    #Thw following will map unvisited nodes to (closest_distance, parent)
    #parent will obviously be a visited node
    node_distances = {}
 
    #Initialization with respect to root
    current_node = root
    visited_node_vectors[root] = model.wv[root]
    unvisited_nodes.remove(root)
    visited_nodes.add(root)
 
    #Build the Mind-Map in n-1 iterations
    for i in range(1, len(nodes)):
        #For every unvisited node 'x'
        for x in unvisited_nodes:
            #Compute contextual distance between current node and x
            dist_from_current = cosine(visited_node_vectors[current_node],
                                       model.wv[x])
            #Get the least contextual distance to x found until now
            distance = node_distances.get(x, (100, ''))
            #If current node provides a shorter path to x, update x's
            #distance and parent information
            if distance[0] > dist_from_current:
                node_distances[x] = (dist_from_current, current_node)
 
        #Choose next 'current' as that unvisited node, which has the
        #lowest contextual distance from any of the visited nodes
        next_node = min(unvisited_nodes,
                        key=lambda x: node_distances[x][0])
 
        ##Update all containers
        parent = node_distances[next_node][1]
        del node_distances[next_node]
        next_node_vect = ((1 - alpha)*model.wv[next_node] +
                          alpha*visited_node_vectors[parent])
        visited_node_vectors[next_node] = next_node_vect
        unvisited_nodes.remove(next_node)
        visited_nodes.add(next_node)
 
        #Add the link between newly selected node and its parent(from the
        #visited nodes) to the NetworkX Graph instance
        g.add_edge((del_ara_link_char(stemmer.original_form(parent).capitalize())),
                   (del_ara_link_char(stemmer.original_form(next_node).capitalize())))

        #The new node becomes the current node for the next iteration
        current_node = next_node
 
    return g


def create_mind_maps(model, root, nodes):
    G = (build_mind_map(model, root, nodes, 0.2))

    # same layout using matplotlib with no labels
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True)

    # Top to Bottom Layout
    A = to_agraph(G)
    A.layout('dot')
    A.draw(path_top_2_bottom_map)

    # Central Layout
    B = to_agraph(G)
    B.layout('twopi')
    B.draw(path_central_map)

def main(txt):    
    word_frequencies_stem, sentence_terms, model = word2vec(txt)

    nodes = get_top_n_terms(word_frequencies_stem, sentence_terms, len(word_frequencies_stem))

    root = max(word_frequencies_stem, key=word_frequencies_stem.get)

    create_mind_maps(model, root, nodes)
