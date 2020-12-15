import nltk
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from newspaper import Article

warnings.filterwarnings('ignore')

article = Article('https://en.wikipedia.org/wiki/Cricket')
article.download()
article.parse()
article.nlp()
raw = article.text
sent_tokens = nltk.sent_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words("english"))


def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens if token not in stop_words]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUT = ("hello", "hai", "greetings", "sup", "what's up", "hey")
GREETING_OUTPUT = ("hi", "hey", "nods", "hi there", "hello")


def greeting(sentence):
    for word in nltk.word_tokenize(sentence):
        if word.lower() in GREETING_INPUT:
            return random.choice(GREETING_OUTPUT)


def response(user_response):
    chat_bot_response = ''
    sent_tokens.append(user_response)
    tf_id_vec = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')
    tf_idf = tf_id_vec.fit_transform(sent_tokens)
    vals = cosine_similarity(tf_idf[-1], tf_idf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tf_idf = flat[-2]
    if req_tf_idf == 0:
        chat_bot_response += "Didn't get you"
    else:
        chat_bot_response += sent_tokens[idx]
    return chat_bot_response


flag = True
print("Start")
while flag:
    user_response = input("type: ")
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("welcome")
        else:
            if greeting(user_response) is not None:
                print("chatbot: ", greeting(user_response))
            else:
                print("chatbot: ", response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("bye")
