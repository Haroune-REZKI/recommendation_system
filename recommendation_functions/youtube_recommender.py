from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

df_cleaned = pd.read_csv('./datasets/youtube/cleaned_course_data.csv')
df = df_cleaned


# Vectorize our Text
count_vect = TfidfVectorizer()
cv_mat = count_vect.fit_transform(df_cleaned['course_title'])

cosine_sim_mat = cosine_similarity(cv_mat)


def preprocess_title(title):
    # Tokenize the title
    tokens = word_tokenize(title)
    # Convert to lower case
    tokens = [token.lower() for token in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens



def recommend_youtube_courses(text,num_of_rec=10):
    def find_best_matches(processed_tokens, df, num_of_rec=10):
        # Combine tokens back to a string
        processed_title = ' '.join(processed_tokens)
        
        # Create a TF-IDF Vectorizer object
        vectorizer = TfidfVectorizer()
        
        # Fit and transform the course titles
        tfidf_matrix = vectorizer.fit_transform(df_cleaned['course_title'])
        
        # Transform the processed title
        processed_tfidf = vectorizer.transform([processed_title])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(processed_tfidf, tfidf_matrix).flatten()
        
        # Get the top matches indices sorted by similarity score
        top_matches_indices = cosine_similarities.argsort()[-num_of_rec:][::-1]

        # Get the top matches similarity scores
        top_matches_scores = cosine_similarities[top_matches_indices]

        # Create a DataFrame with indices and scores
        top_matches_df = pd.DataFrame({'index': top_matches_indices, 'similarity_score': top_matches_scores})
        
        # Merge the original DataFrame with the top matches DataFrame
        merged_df = top_matches_df.merge(df, left_on='index', right_index=True)
        merged_df = merged_df[merged_df['similarity_score']>0.25]
        
        return merged_df.drop(columns=['index'])
    processed_tokens = preprocess_title(text)
    return find_best_matches(processed_tokens,df,num_of_rec=num_of_rec)



