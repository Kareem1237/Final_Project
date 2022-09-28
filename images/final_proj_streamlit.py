import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns 
from wordcloud import STOPWORDS
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
import re
import requests
from sklearn import tree
import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import pycountry
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
import regex as re

import requests
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title('6- Insert a url of an article to check for riskiness 🔍')
link=st.sidebar.text_input('URL','https://seekingalpha.com/news/3885328-ibm-to-acquire-texas-digital-product-development-consultancy?source=content_type%3Areact%7Cfirst_level_url%3Amarket-news%7Csection_asset%3Amain%7Csection%3Atechnology')
df=pd.read_csv(r"./images/cleaned_data.csv")

x=df['cleaned_title']
y=df['Risk']
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)

tree = DecisionTreeClassifier(random_state=10)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)


alpha2=[country.alpha_2 for country in pycountry.countries ]
alpha3=[country.alpha_3 for country in pycountry.countries ]
country_name=[country.name for country in pycountry.countries ]
def clean_up(s):
    """
    Cleans up numbers and special characters from a string.
    
    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    
    
    
    s= re.sub('[0-9]',"", s)
    for x in alpha2:
        s= re.sub(x, "", s)
    for y in alpha3:
        s= re.sub(y, "", s)
    for j in country_name:
        s= re.sub(j, "", s)
    
    s= re.sub('[|](.*)', "", s)
    s= re.sub('[()]', "", s)
    s= re.sub('[/[/]]', "", s)
    s= re.sub("[-,#,',@,.,;,!,?,$,%,-,:,—,&,’,–]", "", s)
    
    s=s.lstrip()
    s=s.rstrip()
    return s

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    return word_tokenize(s)

def remove_cap(s):
    x=[]
    for i in range(len(s)):
        if (s[i].istitle()== False | s[i].isupper()==False) & (len(s[i])>1):
            x.append(s[i])
    return x

def bicap_check(s):
    sum_cap=0
    for i in s:
        if i.isupper():
            sum_cap+=1
    if sum_cap>=2:
        return True
    else:
        return False
def remove_bicap(s):
    x=[]
    for i in range(len(s)):
        if bicap_check(s[i])==False:
            x.append(s[i])
    return x

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    for i in range(len(l)):
        l[i]=ps.stem(l[i])
        l[i]=lemmatizer.lemmatize(l[i])
    return l

stop=STOPWORDS

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    for word in stop:
        while word in l:
            l.remove(word)
    return l

def joining(s):
    s=" ".join(s)
    return s



def clean_all(link):
    url_list=[link]
    for url in url_list:
# making requests instance
        reqs = requests.get(url)
 

        soup = BeautifulSoup(reqs.text, 'html.parser')


        s=soup.find('title').get_text()
   
    s=clean_up(s)
    s=tokenize(s)
    s=remove_cap(s)
    s=remove_bicap(s)
    s=stem_and_lemmatize(s)
    s=remove_stopwords(s)
    s=joining(s)
    s=vectorizer.transform([s])
    s=tree.predict(s)
    if s[0]==0:
        s='Risky ☠️'
    elif s[0]==1:
        s='Safe ✅'
    return s
s=clean_all(link)
st.sidebar.title(s)





iron_hack=Image.open(r".\logo_!.png")




iron_hack=Image.open(r".\logo_!.png")


st.image(iron_hack,width=600)
st.title("A presentation by Karim Majdoub")
st.header("on:")
st.title('Predicting risky investments using article titles')

st.header(" ")
st.header("")
st.header(" ")
st.header("")

st.header(" ")
st.header("")
st.header(" ")
business_case_slide=Image.open(r".\business_case_slide.png")


st.image(business_case_slide,width=1000)

st.header(" ")
st.header("")
st.header(" ")
st.header("")
plan=Image.open(r".\plan.png")


st.image(plan,width=1000)


st.header(" ")
st.header("")
st.header(" ")
st.header("")
st.header('1- Data collection')
st.write('Scrapping titles from article websites')
data_collect=Image.open(r".\get_urls.png")
st.image(data_collect,width=1000)

data_collect=Image.open(r".\data_frame_titles.png")
st.image(data_collect,width=1000)

st.header(" ")
st.header("")
st.header(" ")
st.header("")
st.title('2- Data cleaning')
cleaning_titles_ppt=Image.open(r".\cleaning_titles.png")
st.image(cleaning_titles_ppt,width=1000)



st.header('Clean up')
clean_up=Image.open(r".\clean_up.png")
st.image(clean_up,width=1000)
st.header("")
st.header('Tokenization')
tokenize=Image.open(r".\tokenize.png")
st.image(tokenize,width=1000)
st.header("")    
st.header('Removing caps')
remove_caps=Image.open(r".\remove_caps.png")
st.image(remove_caps,width=1000)
st.header("")    
st.header('Stemming and lemming')
stemm_lemm=Image.open(r".\stemm_lemm.png")
st.image(stemm_lemm,width=1000)
st.header("")    
st.header('Removing stop words and joining')
stop_join=Image.open(r".\stop_join.png")
st.image(stop_join,width=1000)
###########################################################################
alpha2=[country.alpha_2 for country in pycountry.countries ]
alpha3=[country.alpha_3 for country in pycountry.countries ]
country_name=[country.name for country in pycountry.countries ]
def clean_up(s):

    
    
    s= re.sub('[0-9]',"", s)
    for x in alpha2:
        s= re.sub(x, "", s)
    for y in alpha3:
        s= re.sub(y, "", s)
    for j in country_name:
        s= re.sub(j, "", s)
    s= re.sub('stock', "", s)
    s= re.sub('[|](.*)', "", s)
    s= re.sub('[()]', "", s)
    s= re.sub('[/[/]]', "", s)
    s= re.sub("[-,#,',@,.,;,!,?,$,%,-,:,—,&,’,–]", "", s)
    
    s=s.lstrip()
    s=s.rstrip()
    return s

def tokenize(s):

    return word_tokenize(s)

def remove_cap(s):
    x=[]
    for i in range(len(s)):
        if (s[i].istitle()== False | s[i].isupper()==False) & (len(s[i])>1):
            x.append(s[i])
    return x

def bicap_check(s):
    sum_cap=0
    for i in s:
        if i.isupper():
            sum_cap+=1
    if sum_cap>=2:
        return True
    else:
        return False
def remove_bicap(s):
    x=[]
    for i in range(len(s)):
        if bicap_check(s[i])==False:
            x.append(s[i])
    return x

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def stem_and_lemmatize(l):

    for i in range(len(l)):
        l[i]=ps.stem(l[i])
        l[i]=lemmatizer.lemmatize(l[i])
    return l

stop=STOPWORDS

def remove_stopwords(l):

    for word in stop:
        while word in l:
            l.remove(word)
    return l

def joining(s):
    s=" ".join(s)
    return s
st.title('3- EDA')
df=pd.read_csv(r"./sources.csv")
df['title_eda']=df['Title']
df['title_eda']=df['title_eda'].apply(clean_up)
df['title_eda']=df['title_eda'].apply(tokenize)
df['title_eda']=df['title_eda'].apply(remove_cap)
df['title_eda']=df['title_eda'].apply(remove_bicap)
df['title_eda']=df['title_eda'].apply(remove_stopwords)
df['title_eda']=df['title_eda'].apply(joining)
df.drop(df.loc[df['title_eda']==''].index,inplace=True)
risky=df[df['Risk']==0]
safe=df[df['Risk']==1]
df.loc[df['Risk']==0,'Riskiness']='Risky'
df.loc[df['Risk']==1,'Riskiness']='Safe'
risky=df[df['Risk']==0]
pie=df.groupby('Riskiness').agg({'Riskiness':'count'})
pie.rename(columns={'Riskiness':'ratio'},inplace=True)
pie['ratio']=pie['ratio']*100/pie['ratio'].sum()

fig = px.pie(pie, values='ratio', names=pie.index,template='plotly_dark', title='Percentage of risky and safe articles',color=pie.index,color_discrete_map={'Safe':'green',
'Risky':'red'},width=1000,height=700)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(showlegend=False,title = dict(
font = dict(size=24)))
fig.update_traces(textfont_size=20,
                  marker=dict( line=dict(color='white', width=5)))
fig.update_layout(title_font_color="gold",
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
st.write(fig)



common=pd.DataFrame({'Word':list(stop)})
all_words_title = " ".join(word for word in df['Title'])
all_words_title_split=all_words_title.split()
def stop_counter(s):
    tot=all_words_title_split.count(s)
    return tot
common['counter']=common['Word'].apply(stop_counter)
common_sorted=common.sort_values(by='counter',ascending=False)
common_sorted.reset_index(inplace=True,drop=True)

fig = px.bar(common_sorted[:10], x='Word', y='counter',title="Stop words occurunces in the titles",labels={'Word':'Word','counter':'Word occurence'},template='plotly_dark',color='Word',width=800,height=800)
fig.update_xaxes(showgrid=False,color='gold',title_font=dict(size=20))
fig.update_yaxes(showgrid=False,color='gold',title_font=dict(size=20))
fig.update_layout(showlegend=False)
fig.update_layout(xaxis = dict(
tickfont = dict(size=20)),yaxis = dict(
tickfont = dict(size=15)),title = dict(
font = dict(size=24)))
fig.update_traces( marker_line_color='white',
                  marker_line_width=5)
fig.update_layout(title_font_color="gold",
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
st.write(fig)
def stop_word_counter(s):
    count=0
    for word in stop:
         count=count+s.count(word)
    return count
df['tokens']=df['Title'].apply(tokenize)
df['stop word count']=df['tokens'].apply(stop_word_counter)

fig = px.histogram(df, x="stop word count",template='plotly_dark',title='Stop word distribution',labels={'stop word count':'Stop Word Count'},width=800,height=800)
fig.update_xaxes(showgrid=False,color='gold',title_font=dict(size=20))
fig.update_yaxes(showgrid=False,color='gold',title_font=dict(size=20))
fig.update_layout(showlegend=False)
fig.update_layout(xaxis = dict(
tickfont = dict(size=20)),yaxis = dict(
tickfont = dict(size=15)),title = dict(
font = dict(size=24)))
fig.update_traces( marker_line_color='white',
                  marker_line_width=5)
fig.update_layout(title_font_color="gold",
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

st.write(fig)




from collections import Counter
df['tokens_no_stop']=df['Title'].apply(clean_up)
df['tokens_no_stop']=df['tokens_no_stop'].apply(tokenize)
df['tokens_no_stop']=df['tokens_no_stop'].apply(remove_stopwords)
df['tokens_no_stop']=df['tokens_no_stop'].apply(remove_cap)
words = " ".join(w for word in df['tokens_no_stop'] for w in word)
words=words.split()
counter=Counter(words)
non_common=pd.DataFrame({'Word':[keys for keys,values in counter.items()],'Frequency':[values for keys,values in counter.items()]})
non_common_sorted=non_common.sort_values(by='Frequency',ascending=False)
non_common_sorted['ratio']=non_common_sorted['Frequency']*100/non_common_sorted['Frequency'].sum()
non_common_sorted.reset_index(inplace=True)
fig = px.bar(non_common_sorted[:10], x='Word', y='Frequency',hover_data=['ratio'],title="non-Stop words occurunces in all the titles",labels={'Word':'Word','counter':'Word occurence'},template='plotly_dark',color='Word',width=800,height=800)
fig.update_xaxes(showgrid=False,color='gold',title_font=dict(size=20))
fig.update_yaxes(showgrid=False,color='gold',title_font=dict(size=20))
fig.update_layout(showlegend=False)
fig.update_layout(xaxis = dict(
tickfont = dict(size=20)),yaxis = dict(
tickfont = dict(size=15)),title = dict(
font = dict(size=24)))
fig.update_traces( marker_line_color='white',
                  marker_line_width=5)
fig.update_layout(title_font_color="gold",
    title={
        
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
st.write(fig)


risky=df[df['Risk']==0]
safe=df[df['Risk']==1]
#wcloud_words = " ".join(word for word in risky['title_eda'])
st.header('Risky investments word cloud')
#pic = np.array(Image.open(requests.get('http://www.clker.com/cliparts/5/5/d/6/1194989754480445982tiger_graig_ryan_smith_-_01.svg.med.png',stream=True).raw))
# Create a word cloud image
#wordcloud = WordCloud(background_color="white", max_words=1000,
               #contour_width=3, contour_color='black').generate(wcloud_words)
risky_cloud=Image.open(r".\risky_cloud.png")


st.image(risky_cloud,width=600)

st.header(" ")
st.header('Safe investments word cloud')


safe_cloud=Image.open(r".\safe_cloud.png")


st.image(safe_cloud,width=600)




st.header(" ")
st.header(" ")
st.title('4- Building the SQL database')
st.header("")    

sql_comp=Image.open(r".\SQL comp.png")
st.image(sql_comp,width=1000)




st.header("") 
erd_code=Image.open(r".\erd_code.png")
st.image(erd_code,width=1000)




st.header("") 
sql_queries=Image.open(r".\sql_queries.png")
st.image(sql_queries,width=1000)


st.header("") 
st.header("") 

st.title('5- Machine learning')

st.header("") 
ml_class=Image.open(r".\ml_class.png")
st.image(ml_class,width=1000)

st.header("") 
comp_class=Image.open(r".\comp_class.png")
st.image(comp_class,width=1000)













df=pd.read_csv(r".\cleaned_data.csv")

x=df['cleaned_title']
y=df['Risk']
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)

tree = DecisionTreeClassifier(random_state=10)
tree.fit(x_train,y_train)
y_pred=tree.predict(x_test)


alpha2=[country.alpha_2 for country in pycountry.countries ]
alpha3=[country.alpha_3 for country in pycountry.countries ]
country_name=[country.name for country in pycountry.countries ]
def clean_up(s):
    """
    Cleans up numbers and special characters from a string.
    
    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    
    
    
    s= re.sub('[0-9]',"", s)
    for x in alpha2:
        s= re.sub(x, "", s)
    for y in alpha3:
        s= re.sub(y, "", s)
    for j in country_name:
        s= re.sub(j, "", s)
    
    s= re.sub('[|](.*)', "", s)
    s= re.sub('[()]', "", s)
    s= re.sub('[/[/]]', "", s)
    s= re.sub("[-,#,',@,.,;,!,?,$,%,-,:,—,&,’,–]", "", s)
    
    s=s.lstrip()
    s=s.rstrip()
    return s

def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    return word_tokenize(s)

def remove_cap(s):
    x=[]
    for i in range(len(s)):
        if (s[i].istitle()== False | s[i].isupper()==False) & (len(s[i])>1):
            x.append(s[i])
    return x

def bicap_check(s):
    sum_cap=0
    for i in s:
        if i.isupper():
            sum_cap+=1
    if sum_cap>=2:
        return True
    else:
        return False
def remove_bicap(s):
    x=[]
    for i in range(len(s)):
        if bicap_check(s[i])==False:
            x.append(s[i])
    return x

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    for i in range(len(l)):
        l[i]=ps.stem(l[i])
        l[i]=lemmatizer.lemmatize(l[i])
    return l

stop=STOPWORDS

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """
    for word in stop:
        while word in l:
            l.remove(word)
    return l

def joining(s):
    s=" ".join(s)
    return s









