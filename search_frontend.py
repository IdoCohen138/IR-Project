from flask import Flask, request, jsonify
from IR_big_index import *
from google.cloud import storage
import os


K = 0.75
B = 0.40
W_TITLE = 0.6
W_TEXT = 0.333
W_ANCHOR = 0.6
TOP_HUNDRED = 100


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        """
            initillize some attributes to the app :

            self.client -
            self.bucket -  are attributes for link to the gcp

            self.index_text_wo_stemm - body index without stemming
            self.index_text - body index with stemming
            self.index_tite - title index
            self.index_anchor - anchor_text index

            self.pr - dictionary for PAGERANK
            self.pv - dictionary for PAGEVIEWS
            self.dict_title - dictionary that mapping doc_id to title

        """


        self.client = storage.Client()
        self.bucket = self.client.bucket('irproject315450569')

        blobreader = self.bucket.get_blob(f"big_index_text_w_no_stemm{os.sep}index_w_no_stemm.pkl").open('rb')
        self.index_text_wo_stemm = pickle.load(blobreader)
        blobreader.close()

        blobreader = self.bucket.get_blob(f"big_index_text{os.sep}index.pkl").open('rb')
        self.index_text = pickle.load(blobreader)
        blobreader.close()

        self.index_title = InvertedIndex.read_index(f'big_index_title{os.sep}', 'big_index_title' )
        self.index_anchor = InvertedIndex.read_index(f'big_index_anchor{os.sep}', 'big_index_anchor')

        self.pr = {}
        with (open(f'dicts{os.sep}dict_pr.pkl', "rb")) as openfile:
            while True:
                try:
                    self.pr.update(pickle.load(openfile))
                except EOFError:
                    break

        self.pv = {}
        with (open(f'dicts{os.sep}dict_pv.pkl', "rb")) as openfile:
            while True:
                try:
                    self.pv.update(pickle.load(openfile))
                except EOFError:
                    break

        self.dict_title = {}
        with (open(f'dicts{os.sep}dict_title.pkl', "rb")) as openfile:
            while True:
                try:
                    self.dict_title.update(pickle.load(openfile))
                except EOFError:
                    break

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    normalized_factor_pr = app.pr[list(app.pr.keys())[0]]

    query_clear = get_q_after_tok_stem(query)

    dict_BM25_text = calculate_BM25(query_clear, app.index_text, K, B)
    dict_binary_title = effective_cosine_binary_title(query_clear, app.index_title)
    dict_binary_anchor = effective_cosine_binary_anchor(query_clear,app.index_anchor)

    result = merge_3_results(dict_binary_title,dict_BM25_text, dict_binary_anchor, W_TITLE,W_TEXT,W_ANCHOR)

    res = [(doc_id, app.dict_title[doc_id]) for doc_id, score in result.items()][:TOP_HUNDRED]

    multiply = []
    for doc_id, title in res:
        score = result[doc_id]
        try:
            new_score = app.pr[doc_id]*(1/normalized_factor_pr) + score
        except:
            new_score = score
        multiply.append((doc_id,new_score))

    sort = sorted(multiply, key=lambda x: x[1], reverse=True)

    res = [(doc_id, app.dict_title[doc_id]) for doc_id, score in sort]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    dict_TFIDF_text = effective_cosine(query, app.index_text_wo_stemm)

    res = [(doc_id, app.dict_title[doc_id]) for doc_id, score in dict_TFIDF_text.items()][:TOP_HUNDRED]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_clear = get_q_after_tok_stem(query)

    dict_binary_title = effective_cosine_binary_title(query_clear, app.index_title)

    res = [(doc_id, app.dict_title[doc_id]) for doc_id, score in dict_binary_title.items()]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    query_clear = get_q_after_tok_stem(query)

    dict_binary_anchor = effective_cosine_binary_anchor(query_clear, app.index_anchor)

    res = [(doc_id, app.dict_title[doc_id]) for doc_id, score in dict_binary_anchor.items()]

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    for i in range(len(wiki_ids)):
        try:
            score = app.pr[wiki_ids[i]]
            res.append(score)
        except:
            res.append(0)

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    for i in range(len(wiki_ids)):
        try:
            score = app.pv[wiki_ids[i]]
            res.append(score)
        except:
            res.append(0)

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080)
