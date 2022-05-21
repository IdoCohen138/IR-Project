from inverted_index_gcp import *
from collections import Counter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
import numpy as np
import math
import struct

AVGDL = 320.405
N = 6348910
TUPLE_SIZE_ANCHOR_TITLE = 4
TUPLE_SIZE = 16

from inverted_index_gcp_title import MultiFileReader as inverted_title_reader
from inverted_index_gcp_anchor import MultiFileReader as inverted_anchor_reader
from inverted_index_gcp import MultiFileReader as inverted_text_reader
from inverted_index_gcp_wo_stemm import MultiFileReader as inverted_text_wo_stemm_reader


def read_pl_binary_title(inverted, w):
  """
   read posting lists from title index
  """
  with closing(inverted_title_reader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE_ANCHOR_TITLE)
    posting_list = []
    for i in range(inverted.df[w]):
      posting_list.append(struct.unpack("I", b[i*TUPLE_SIZE_ANCHOR_TITLE:(i+1)*TUPLE_SIZE_ANCHOR_TITLE]))
    return posting_list

def read_pl_binary_anchor(inverted, w):
  """
   read posting lists from anchor index
  """
  with closing(inverted_anchor_reader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE_ANCHOR_TITLE)
    posting_list = []
    for i in range(inverted.df[w]):
      posting_list.append(struct.unpack("I", b[i*TUPLE_SIZE_ANCHOR_TITLE:(i+1)*TUPLE_SIZE_ANCHOR_TITLE]))
    return posting_list

def read_pl_text(inverted, w):
  """
  read posting lists from text index
  """
  with closing(inverted_text_reader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      posting_list.append(struct.unpack("IHfI", b[i*TUPLE_SIZE:(i+1)*TUPLE_SIZE]))
    return posting_list

def read_pl_text_wo_stemm(inverted, w):
  """
     read posting lists from text index without stemming
  """
  with closing(inverted_text_wo_stemm_reader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
        posting_list.append(struct.unpack("IHfI", b[i*TUPLE_SIZE:(i+1)*TUPLE_SIZE]))
    return posting_list

def get_q_after_tok_stem(query):
    """
     function for clean the query and stemming
    """
    tokens = [token.group() for token in RE_WORD.finditer(query.lower()) if
                    token.group() not in english_stopwords]
    ps = PorterStemmer()
    result= [ps.stem(word) for word in tokens]

    return result


def calculate_BM25(query_clear, index, k, b):
    """
     calculate BM25 for "search" function
    """
    counter_q = Counter(query_clear)
    results = {}
    for term in query_clear:
        try:
            posting_list = read_pl_text(index, term)
        except:
            continue

        for doc_id, tf, tf_idf_normalize, doc_len in posting_list:
            if doc_id not in results.keys():
                results[doc_id] = 0

            mone = counter_q[term] * (k + 1) * tf
            mehane = tf + k * (1 - b + b * doc_len / AVGDL)
            results[doc_id] += mone / mehane * math.log((1 + N) / index.df[term])

    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    return results


def generate_query_tfidf_vector(query_to_search, index):
  """
  Generate a vector representing the query. Each entry within this vector represents a tfidf score.
  The terms representing the query will be the unique terms in the index.

  We will use tfidf on the query as well.
  For calculation of IDF, use log with base 10.
  tf will be normalized based on the length of the query.

  Parameters:
  -----------
  query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                   Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

  index:           inverted index loaded from the corresponding files.

  Returns:
  -----------
  dictionary of query with tfidf scores
  """
  epsilon = .0000001
  Q = {}

  counter = Counter(query_to_search)

  for token in np.unique(query_to_search):
      if token in index.df.keys():  # avoid terms that do not appear in the index.
          tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
          df = index.df[token]
          idf = math.log2(N / (df + epsilon))  # smoothing

          try:
              # ind = term_vector.index(token)
              Q[token] = tf * idf
          except:
              pass
  return Q


def effective_cosine(query, index):
  """
     calculate cosine similarity for 'search_body' function
  """
  sim_dict = {} # dictionary in the form : { doc_id : [similarity_cosine, sumofweights**2] }

  query_tok = [token.group() for token in RE_WORD.finditer(query.lower()) if
            token.group() not in english_stopwords]

  Q = generate_query_tfidf_vector(query_tok, index)

  weight_t_in_querie = 0

  for term in query_tok:
    if term not in Q.keys():
        Q[term] = 0
    weight_t_in_querie += math.pow(Q[term], 2)
    try:
      posting_list = read_pl_text_wo_stemm(index, term)
    except:  
      continue
    for tup in posting_list: # tup[0] = doc_id, tup[1] = tf, tup[2] = tfidf/normalized_factor , tup[3] = len(doc)
      if (tup[0] not in sim_dict.keys()):
        sim_dict[tup[0]] = 0
      sim_dict[tup[0]] += Q[term] * tup[2]

  #normalize:
  for key in sim_dict.keys():
    sim_dict[key] = sim_dict[key] * (1/math.sqrt(weight_t_in_querie))

  sim_dict = {k : v for k, v in sorted(sim_dict.items(), key=lambda item : item[1], reverse=True)}

  return sim_dict

def merge_3_results(title_scores, body_scores, anchor_scores, title_weight, text_weight, anchor_weight):
    """
    merging 3 results from 3 indecies
    """
    result = {}
    checked_doc = {}
    # merge
    if len(title_scores) == 0 and len(body_scores) == 0:
        return anchor_scores
    if len(body_scores) == 0 and len(anchor_scores) == 0:
        return title_scores
    if len(title_scores) == 0 and len(anchor_scores) == 0:
        return body_scores

    for doc_id, score in title_scores.items():
        result[doc_id] = title_weight * title_scores[doc_id]

        if doc_id in body_scores.keys():  # if yes - we will search him
            result[doc_id] += text_weight * body_scores[doc_id]
        if doc_id in anchor_scores.keys():
            result[doc_id] += anchor_weight * anchor_scores[doc_id]

        checked_doc[doc_id] = 1

    for doc_id in body_scores.keys():
        if (doc_id not in checked_doc.keys()):
            result[doc_id] = text_weight * body_scores[doc_id]
            if doc_id in anchor_scores.keys():
                result[doc_id] += anchor_weight * anchor_scores[doc_id]

            checked_doc[doc_id] = 1

    for doc_id in anchor_scores.keys():
        if (doc_id not in checked_doc.keys()):
            result[doc_id] = anchor_weight * anchor_scores[doc_id]

            checked_doc[doc_id] = 1

    # sort
    result = {k : v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}

    return result

def effective_cosine_binary_anchor(query_clear, index):
  """
  calculate binary similarity for anchor text
  """
  sim_dict = {}  # dictionary in the form : { doc_id : [similarity_cosine , lenDoc] }

  for term in query_clear:
      try:
          posting_list = read_pl_binary_anchor(index, term)
      except:
          continue
      for tup in posting_list:  # tup[0] = term, tup[1] = tfidf, tup[2] = lenDOC
          if (tup[0] not in sim_dict.keys()):
              sim_dict[tup[0]] = 0
          sim_dict[tup[0]] += 1

  for key in sim_dict.keys():
      sim_dict[key]= sim_dict[key] * (1 / len(query_clear))

  sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}

  return sim_dict

def effective_cosine_binary_title(query_clear, index):
  """
  calculate binary similarity for title text
  """
  sim_dict = {}  # dictionary in the form : { doc_id : [similarity_cosine , lenDoc] }

  for term in query_clear:
      try:
          posting_list = read_pl_binary_title(index, term)
      except:
          continue
      for tup in posting_list:  # tup[0] = term, tup[1] = tfidf, tup[2] = lenDOC
          if (tup[0] not in sim_dict.keys()):
              sim_dict[tup[0]] = 0
          sim_dict[tup[0]] += 1

  for key in sim_dict.keys():
      sim_dict[key]= sim_dict[key] * (1 / len(query_clear))

  sim_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1], reverse=True)}

  return sim_dict
