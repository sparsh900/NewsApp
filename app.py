from sre_parse import Tokenizer
from unicodedata import category
from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)
ps = PorterStemmer()
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import sent_tokenize
import numpy as np
import networkx as nx
import re
from newsapi import NewsApiClient
from datetime import datetime

# Init
newsapi = NewsApiClient(api_key='55bba2d308d647759e0763df9ef309b0')

top_headlines = newsapi.get_top_headlines(country='us')

# entertainment = newsapi.get_top_headlines(category='entertainment')

# business = newsapi.get_top_headlines(category='business')

# health = newsapi.get_top_headlines(category='health')

# sports = newsapi.get_top_headlines(category='sports')

#print(top_headlines['articles'][0]['publishedAt'].strptime('%Y-%m-%d'))
# print(datetime.strptime(top_headlines['articles'][0]['publishedAt'], '%Y%m%d%H%M%S').strftime('%Y-%m-%d'))

#Function to split text into sentences by fullstop(.)
'''def read_article(text):
    
    article = text.split(". ")
    sentences =[]
    
    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
    
    return sentences'''

# Read the text and tokenize into sentences
def read_article(text):
    
    sentences =[]
    
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence.replace("[^a-zA-Z0-9]"," ")

    return sentences
    

# Create vectors and calculate cosine similarity b/w two sentences
def sentence_similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    #build the vector for the first sentence
    for w in sent1:
        if not w in stopwords:
            vector1[all_words.index(w)]+=1
    
    #build the vector for the second sentence
    for w in sent2:
        if not w in stopwords:
            vector2[all_words.index(w)]+=1
            
    return 1-cosine_distance(vector1,vector2)

# Create similarity matrix among all sentences
def build_similarity_matrix(sentences,stop_words):
    #create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1!=idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stop_words)
                
    return similarity_matrix


# Generate and return text summary
def generate_summary(text,top_n):
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    stop_words = stopwords.words('english')
    summarize_text = []
    
    # Step1: read text and tokenize
    sentences = read_article(text)
    
    # Steo2: generate similarity matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences,stop_words)
    
    # Step3: Rank sentences in similarirty matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    #Step4: sort the rank and place top sentences
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    
    # Step 5: get the top n number of sentences based on rank    
    for i in range(top_n):
        summarize_text.append(ranked_sentences[i][1])
    
    # Step 6 : outpur the summarized version
    return " ".join(summarize_text),len(sentences)


# @app.route('/summary/', methods=['POST'])
# def weba():
#     text = request.form['text']
#     summary, original_length = generate_summary(text,4)
        
#     return render_template('index.html',
#                                result_summ=summary,
#                                text_summ = text,
#                                lines_summary = 3)




# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidfvect = pickle.load(open('tfidfvect.pkl', 'rb'))
# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', top_headlines=top_headlines)
def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    summary, original_length = generate_summary(text,3)
        
    return render_template('index.html',text=text, result=prediction,
                               result_summ=summary,
                               lines_summary = 3,
                               top_headlines=top_headlines
                               )
    # return render_template('index.html', text=text, result=prediction)
@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
if __name__ == "__main__":
    app.run()