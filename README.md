# IR-Project

> This repository doesn't contain the index files, which need to be created in the GCP using the relevant code. It also doesn't include the general dictionaries (doc id to title, doc id to page views, doc id to page rank and doc id to combined page rank and page views, which also need to be created using the relevant code - on the local machine 

This project implements a search engine one the whole english wikipedia corpus. Given a query, it finds the most relevant documents. The query could be any free text, to be searched on the body/title/anchor of the articles, or on the combination of them. It also can support queries, that include doc id, and asks for the page rank and page views of the documents. The service is ran by a flask server, which is deployed on a google cloud VM instance. The server supports 6 different http requests on port 8080: search, search_body, search_anchor, search_title, get_pageview and get_pagerank. The engine clears the query, tokenizes and stems it, addes 2 word n-grams to it, and finally searches the relevant documents in the inverted index - which contains a score for each combination of a word and document, using the bm25 algorithm. In the calculation, the page views of each document are also considered.

## Inverted Index Structure
To access the inverted index file, first the "index.pkl" should be read with pickle. This file contains all the global data of the index:

* AVGDL : Average document length of the whole corpus
* posting_locs: A dictionary, which maps each word to name of the correlated bin file - which contains the whole posting list, and to the offset of the posting list in the file.
* df: A dictionary, which maps each word to the document frequence of each word. The index also contain the bin files - which hold the binary data each word posting list.

## Posting List Structure

### Small Indexes
  Each posting list contains a list of tuples, where each tuple has 4 elements:

* doc_id: The id of the document in wikipedia.
* tf: The term frequence of the term in the document.
* max_tf: The max_tf of the whole document.
* doc_lent: The length of the document. These elements are used for calculations and experiments

### Title and Anchor indexes
  Each posting list contains a list of integers, where each integer is a document id.

### TF-IDF Body Index and BM25 Merged Index:
  Each posting list contains a list of tuples, where each tuple has 2 elements:

* doc_id: The id of the document in wikipedia.
* score: The score of the word in the specific document, calculated by TF-IDF/BM25 formula.

## Project Structure

### Testing&Resualts
  This folder contains the results of the testing of the machine.
### Inverted Index
  This folder contains needed files to create all the big indexes (on the whole corpus), which are finally used to build the engine.
### search_frontend.py
This file contains the flask app which recieves http requests for queries, and returns the results for them.
