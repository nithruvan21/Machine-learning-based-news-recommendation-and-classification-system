import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


#URL of the news website
url = 'https://timesofindia.indiatimes.com/home/headlines?from=mdr'

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the elements containing news headlines
headlines = soup.find_all('span', class_='w_tle')  # adjust the class or tag as per the website's structure

scrapedData=[]
fieldname=["title"]

# Extract and print the headlines
for headline in headlines:
    #print(headline.text.strip())  # .strip() removes any leading/trailing whitespace
    scrapedData.append({'title':headline.text.strip()})

#print(scrapedData)

csv_file = "C:/News.csv"

with open(csv_file, mode='w', newline='', encoding='utf-8')as file:
    writer = csv.DictWriter(file, fieldnames=fieldname)
    writer.writeheader()
    for data in scrapedData:
        writer.writerow(data)

print("CSV file updated!")

news_df = pd.read_csv(csv_file)
#print(news_df.info)

news_df = pd.DataFrame(news_df, columns=['title'])

def create_soup(x):
    soup = ''.join(x['title'])
    return soup

news_df['soup'] = news_df.apply(create_soup, axis=1)

#print(news_df)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(news_df['title'])

print("Stop words....")
print(tfidf.get_stop_words()) #stopwords that are removed from the document
print("TFIDF...")
print(tfidf)
print("TFIDF MATRIX...")
print(tfidf_matrix)
print("Feature names...")
print(tfidf.get_feature_names_out()[500:510]) #feature words that are used for tf

cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix,True)

# display(cosine_sim.shape)
display(cosine_sim)

metadata = news_df.reset_index()
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
#display(indices[:10])

def get_recommendations(title, indices, cosine_sim, data):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:5]
    news_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[news_indices]

header = st.container()
subheader = st.container()

with header:
    st.title("News Recommendation System")
    value = st.text_input("Search: ","India")

word = value
title = word 
pattern = f'\\b{word}\\b'
sentences = metadata[metadata['title'].str.contains(pattern,case=False)]

try:
    title = sentences.iloc[0]['title']
    recommendations = get_recommendations(title, indices, cosine_sim, metadata)
    news = []
    for i in range(5):
        news.append(recommendations.iloc[i])
except IndexError:
    news = []

print(news)

try:
    with subheader:
        if len(news)>0:
            for i in range(5):
                st.subheader(news[i])
        else:
            st.error("No recommendations for this topic. Kindly re-enter a different topic.")
except IndexError:
    with subheader:
        st.subheader("Enter a different topic!!")