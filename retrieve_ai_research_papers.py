
# !pip install nltk gensim

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return lemmatized_tokens

def process_paper(title, abstract):
    # Concatenate title and abstract
    paper_text = title + " " + abstract

    # Preprocess text
    processed_text = preprocess_text(paper_text)

    return processed_text

def extract_keywords(processed_text, num_keywords=5):
    # Calculate word frequency
    fdist = FreqDist(processed_text)

    # Get top N most common words
    top_keywords = fdist.most_common(num_keywords)

    return [keyword[0] for keyword in top_keywords]

# Example research paper data
title = "Deep Learning Techniques for Image Recognition"
abstract = "This paper explores various deep learning models for image recognition tasks, including convolutional neural networks (CNNs) and recurrent neural networks (RNNs). We conduct experiments on benchmark datasets to evaluate the performance of different architectures."

# Process paper
processed_paper = process_paper(title, abstract)

# Extract keywords
keywords = extract_keywords(processed_paper)
print("Keywords:", keywords)

# Example test data
test_data = [
    {
        "title": "Understanding LSTM Networks",
        "abstract": "This paper introduces the Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), and explains their architecture and functioning."
    },
    {
        "title": "Generative Adversarial Networks",
        "abstract": "This paper presents Generative Adversarial Networks (GANs), a class of deep learning models used for generating realistic data samples."
    }
]

# Test the model
for paper in test_data:
    title = paper["title"]
    abstract = paper["abstract"]

    processed_paper = process_paper(title, abstract)
    keywords = extract_keywords(processed_paper)

    print("Title:", title)
    print("Abstract:", abstract)
    print("Extracted Keywords:", keywords)
    print()