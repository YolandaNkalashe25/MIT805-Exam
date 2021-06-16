#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import seaborn as sns
import streamlit as st 
import altair as alt
import base64
import pickle
import joblib
import emoji
from sklearn import preprocessing
import pandas as pd # to read csv/excel formatted data
import matplotlib.pyplot as plt # to plot graphs
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import nltk
import re
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import math
import datetime
from datetime import datetime, timezone
from googletrans import Translator
translator = Translator()
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import pickle
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.utils import simple_preprocess 
from gensim.models import CoherenceModel

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords

def preprocess(data_heading):
        null_dict={}
        nulls_all=data_heading.isnull().sum().to_frame()
        for index, row in nulls_all.iterrows():
          if row[0]>0:
          #print(index, row[0])
            null_dict[index]=row[0]
        nulls_frame = pd.DataFrame(null_dict.items(),columns=['tweet_head','nul'])

        nulls_frame['nul_perc']= (nulls_frame['nul']/len(data_heading['input_query']))*100 # convert sum of null values into %
        nul_col=[] # To store  columns name
        nul_percent=[] # To store percentages

        for i,j in zip(nulls_frame['nul_perc'],nulls_frame['tweet_head']):
          if i>=75: 
            nul_col.append(j)
            nul_percent.append(i)
            del data_heading[j]
        data_non_eng=data_heading[data_heading['statuses_metadata_iso_language_code']!='en']
        data_en=data_heading[data_heading['statuses_metadata_iso_language_code']=='en']

        lang_code=[] # to store detected language iso code
        for i in data_non_eng['statuses_text']:

          try:
            lang = detect(i) # performing language detection
          except:
            lang="und" # put error if language is undetected/or cell is empty[any case]
          
          lang_code.append(lang)

        data_non_eng['LangDet_code']=lang_code #Create a column of the newly detected language iso code

      #Check if the detected iso code are the same as the iso code on the data
        data_non_eng['Iso_Code_Comparison'] = np.where((data_non_eng['LangDet_code'] == data_non_eng['statuses_metadata_iso_language_code']), 'Yes', 'No')

        data_translated=data_non_eng[data_non_eng['LangDet_code']!='und'] #Exclude undetected
        data_translated=data_translated[data_translated['LangDet_code']!='en'] #Excluded detected as English

        data_translate_en=data_non_eng[data_non_eng['LangDet_code']=='en'] # Creat a dataframe of English detected languages [ to be merged with the other data after translation]
        data_translated=data_translated.reset_index(drop=True) #reset index

        translated_text=[] #To store translated text
        for i in data_translated['statuses_text'].index:
          to_translate = data_translated['statuses_text'].iloc[i]
          #translated="text_trans"
          translated = GoogleTranslator(source=data_translated['LangDet_code'].iloc[i], target='en').translate(to_translate) #Perform translation
          translated_text.append(translated) #Append translated text

        data_translated['Translated_tweet']=translated_text # Translated text Column 
        data_translated['statuses_text']=data_translated['Translated_tweet'].values # Overide non-english tweets with translated tweets
        data_translated['statuses_metadata_iso_language_code']=data_translated['LangDet_code'].values #overide iso code with the detected iso code

        del data_translated['Translated_tweet'] # delete additional column[since it has overriden 'statuses_text']
        Translated_final=data_translated.append(data_translate_en,ignore_index = True) # Join English detected and Translated datasets

      #delete duplicating and non used columns on the dataset
        del Translated_final['Iso_Code_Comparison'] #delete added column from the dataset
        del Translated_final['LangDet_code'] #delete added column from the dataset

        Final_Dataset=data_en.append(Translated_final,ignore_index = True) # Join Originally English identified, Translated datasets and English detected Datasets

        return Final_Dataset


def CategoriseSA(Final_Dataset):
        Final_Dataset['statuses_text'] = Final_Dataset['statuses_text'].str.lower()
        Categorisation_dataset=Final_Dataset[(Final_Dataset['input_query']!='nfsas') & (Final_Dataset['input_query']!='#openthechurches')]

        HashTag_Covid=Categorisation_dataset[Categorisation_dataset['input_query']=='Covid']
        HashTag_Vaccine=Categorisation_dataset[Categorisation_dataset['input_query']=='vaccine']
        HashTag_SA=Categorisation_dataset[(Categorisation_dataset['input_query']=='#southafrica') | (Categorisation_dataset['input_query']=='South Africa')|(Categorisation_dataset['input_query']=='#SAlockdown')]

        C=HashTag_Covid['statuses_text'].str.contains('covid |vaccine| pandemic | corona| virus') 
        Hash_Covid=HashTag_Covid[C]

        V=HashTag_Vaccine['statuses_text'].str.contains('covid |vaccine| pandemic | corona| virus') 
        Hash_Vac=HashTag_Vaccine[V]

        S=HashTag_SA['statuses_text'].str.contains('covid |vaccine| pandemic | corona| virus') 
        Hash_SA=HashTag_SA[S]

        Hash_Relevant=Final_Dataset[(Final_Dataset['input_query']=='#covidvaccine') | (Final_Dataset['input_query']=='#VaccineforSouthAfrica')|(Final_Dataset['input_query']=='#VaccineRolloutSA')|(Final_Dataset['input_query']=='#vaccineSA')|(Final_Dataset['input_query']=='vaccine AND "South Africa"')]

        Hash_Vac=Hash_Vac.reset_index(drop=True)
        Hash_Covid=Hash_Covid.reset_index(drop=True)
        Hash_SA=Hash_SA.reset_index(drop=True)

        All_Covid_tweets=Hash_Relevant.append([Hash_SA,Hash_Vac,Hash_Covid],ignore_index=True)

        All_Covid_tweets['statuses_text'] = All_Covid_tweets['statuses_text'].str.replace(r'[^\w\s]+', '')

        All_Covid_tweets['statuses_text'] = All_Covid_tweets['statuses_text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
        All_Covid_tweets['statuses_text'] = All_Covid_tweets['statuses_text'].str.lower()

        from nltk.corpus import stopwords

        stop = stopwords.words('english')
        newStopWords = ['RT','rt','capricornfmnews']
        stop.extend(newStopWords)
        All_Covid_tweets['statuses_without_stopwords']=All_Covid_tweets['statuses_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        A=All_Covid_tweets['statuses_text'].str.contains('south africa|southafrica|ramaphosa|mzansi|cyril|zuma|nzimande|eff|anc|zwelimkhize|mkhize|drzwelimkhize') 
        SA_tweets=All_Covid_tweets[A]
        global_tweets=All_Covid_tweets[~A]

        SA_tweets['Class']=1
        global_tweets['Class']=0

        S=SA_tweets[['statuses_without_stopwords','Class']]
        G=global_tweets[['statuses_without_stopwords','Class']]

      #Data_Models=Data_Models.replace(r"_", "", regex=True)
        Data_Models=S.append(G,ignore_index=True)

        documents = []
        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()
        for tex in range(0, len(Data_Models)):
          # Remove all the special characters
            document = re.sub(r'\W', ' ', str(Data_Models.statuses_without_stopwords[tex]))
          
          # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
          
          # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
          
          # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)    
          # Lemmatization
            document = document.split()

            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
          
            documents.append(document)

      #dataframe of clean text
        df_lem = pd.DataFrame(documents, columns=["clean_text"])
        Data_Models['clean_text']=df_lem['clean_text']

        return  Data_Models #best_model

  
def clean_text(text):
        documents = []
        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()
        for tex in range(0, len(text)):
          # Remove all the special characters
            document = re.sub(r'\W', ' ', str(text[tex]))
          
          # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
          
          # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
          
          # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)    
          # Lemmatization
            document = document.split()

            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
          
            documents.append(document)
      
      #dataframe of clean text
        df_lem = pd.DataFrame(documents, columns=["clean_text"])
        return df_lem

def main():
                
                def Bulk_data(data_load):
                    if data_load is not None:
                        #data = pd.read_excel(data_load)
                        Label_list=['input_query','statuses_created_at','statuses_id','statuses_text','statuses_truncated','statuses_entities_user_mentions[0]_screen_name','statuses_entities_user_mentions[0]_name','statuses_entities_user_mentions[0]_id','statuses_entities_user_mentions[0]_id_str','statuses_entities_user_mentions[0]_indices[0]','statuses_metadata_iso_language_code','statuses_metadata_result_type','statuses_source','statuses_in_reply_to_status_id','statuses_in_reply_to_status_id_str','statuses_in_reply_to_user_id','statuses_in_reply_to_user_id_str','statuses_in_reply_to_screen_name','statuses_user_id','statuses_user_id_str','statuses_user_name','statuses_user_screen_name','statuses_user_location','statuses_user_description','statuses_user_url','statuses_user_entities_url_urls[0]_url','statuses_user_entities_url_urls[0]_expanded_url','statuses_user_entities_url_urls[0]_display_url','statuses_user_entities_url_urls[0]_indices[0]','statuses_user_entities_description_urls[0]_url','statuses_user_entities_description_urls[0]_expanded_url','statuses_user_entities_description_urls[0]_display_url','statuses_user_entities_description_urls[0]_indices[0]','statuses_user_protected','statuses_user_followers_count','statuses_user_friends_count','statuses_user_listed_count','statuses_user_created_at','statuses_user_favourites_count','statuses_user_statuses_count','statuses_user_profile_background_color','statuses_user_profile_background_image_url','statuses_user_profile_background_image_url_https','statuses_user_profile_background_tile','statuses_user_profile_image_url','statuses_user_profile_image_url_https','statuses_user_profile_banner_url','statuses_user_profile_link_color','statuses_user_profile_sidebar_border_color','statuses_user_profile_sidebar_fill_color','statuses_user_profile_text_color','statuses_user_profile_use_background_image','statuses_user_has_extended_profile','statuses_user_default_profile','statuses_user_default_profile_image','statuses_retweeted_status_created_at','statuses_retweeted_status_id','statuses_retweeted_status_id_str','statuses_retweeted_status_text','statuses_retweeted_status_truncated','statuses_retweeted_status_entities_urls[0]_url','statuses_retweeted_status_entities_urls[0]_expanded_url','statuses_retweeted_status_entities_urls[0]_display_url','statuses_retweeted_status_entities_urls[0]_indices[0]','statuses_retweeted_status_metadata_iso_language_code','statuses_retweeted_status_metadata_result_type','statuses_retweeted_status_source','statuses_retweeted_status_user_id','statuses_retweeted_status_user_id_str','statuses_retweeted_status_user_name','statuses_retweeted_status_user_screen_name','statuses_retweeted_status_user_location','statuses_retweeted_status_user_description','statuses_retweeted_status_user_url','statuses_retweeted_status_user_entities_url_urls[0]_url','statuses_retweeted_status_user_entities_url_urls[0]_expanded_url','statuses_retweeted_status_user_entities_url_urls[0]_display_url','statuses_retweeted_status_user_entities_url_urls[0]_indices[0]','statuses_retweeted_status_user_protected','statuses_retweeted_status_user_followers_count','statuses_retweeted_status_user_friends_count','statuses_retweeted_status_user_listed_count','statuses_retweeted_status_user_created_at','statuses_retweeted_status_user_favourites_count','statuses_retweeted_status_user_utc_offset','statuses_retweeted_status_user_verified','statuses_retweeted_status_user_statuses_count','statuses_retweeted_status_user_contributors_enabled','statuses_retweeted_status_user_is_translator','statuses_retweeted_status_user_is_translation_enabled','statuses_retweeted_status_user_profile_background_color','statuses_retweeted_status_user_profile_background_image_url','statuses_retweeted_status_user_profile_background_image_url_https','statuses_retweeted_status_user_profile_background_tile','statuses_retweeted_status_user_profile_image_url','statuses_retweeted_status_user_profile_image_url_https','statuses_retweeted_status_user_profile_banner_url','statuses_retweeted_status_user_profile_link_color','statuses_retweeted_status_user_profile_sidebar_border_color','statuses_retweeted_status_user_profile_sidebar_fill_color','statuses_retweeted_status_user_profile_text_color','statuses_retweeted_status_user_profile_use_background_image','statuses_retweeted_status_user_has_extended_profile','statuses_retweeted_status_user_default_profile','statuses_retweeted_status_user_default_profile_image','statuses_retweeted_status_retweet_count','statuses_retweeted_status_favorite_count','statuses_retweeted_status_favorited','statuses_retweeted_status_retweeted','statuses_retweeted_status_possibly_sensitive','statuses_retweeted_status_lang','statuses_is_quote_status',	'statuses_retweet_count',	'statuses_favorite_count','statuses_favorited',	'statuses_retweeted','statuses_lang']
                        data_load.columns=Label_list
                        #predata=self.preprocess(data)  
                        return data_load
                url='https://github.com/YolandaNkalashe25/COS802-Project/blob/main/EDA_sample4%20(1).xlsx?raw=true'
                #url="https://github.com/AndaniMadodonga/Test/blob/main/Tweetdatatest%20-%20Copy.xlsx?raw=true"
                
                Data_file = pd.read_excel(url)
                
                html_temp1 = """ 
                           <div style ="background-color:yellow;padding:13px"> 
                          <h1 style ="color:black;text-align:Center;;">Visualization/Dashboard based on Trained data</h1> 
                          </div> 
                           """
                if Data_file is None:
                   st.write("Check the data reference link ")
                else:
                  st.markdown(html_temp1, unsafe_allow_html=True)
                   #Full_Data().main_full()
                   
                  data=Bulk_data(Data_file)

                  data_processed=data
                  
                  st.write(data_processed.head())
                  
                  data_cat=pd.DataFrame(data_processed['input_query'])

                  def hastag(row):
                      if(row['input_query']=='vaccine AND "South Africa"' or row['input_query']=='#vaccineSA' or row['input_query']=='#covidvaccine' or row['input_query']=='#VaccineforSouthAfrica' or row['input_query']=='#VaccineRolloutSA' or row['input_query']=='vaccine' ):
                        val="#T_Vaccine"
                      elif(row['input_query']=='"South Africa"' or row['input_query']=='#southafrica' or row['input_query']=='#SAlockdown'):  
                        val="#T_SA"
                      elif(row['input_query']=='Covid' or row['input_query']=='covid' or row['input_query']=='#openthechurches'):  
                        val="#T_Covid19"
                      else:
                        val="Other"
        
                      return val
                   
                  data_cat['input_query']=data_processed.apply(hastag,axis=1)
                    
                                     
                  col1, col2 = st.beta_columns(2)
                  with col1:

                   size=len(data_processed['statuses_retweeted_status_id'].unique())
                
                   st.write(size)
                   st.write('Count of unique tweets')
                   st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
                  with col2:
                    st.write('Count of unique tweets')
                    st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
                      

                  my_expander = st.beta_expander("Show Category and SA Catogory visual", expanded=True)
                  with my_expander:
                  #if st.checkbox('Show Category and SA Catogory visual'):    
 
                   st.subheader( "**Hash_Tags vs Topics Bar Graph**") 
                   sns.set(rc={"figure.figsize":(10,5)})
                   
                   import matplotlib.pyplot as plt
                   
                   # PLotting category input query
                   
                   fig= plt.figure()
                   ax = sns.countplot(y="input_query", data=data_cat,order=data_cat['input_query'].value_counts().index)
                   plt.xticks(rotation=45)
# =============================================================================
#  
#                    for p in ax.patches:
#                          height = p.get_height() 
#                          width = p.get_width() 
#                          ax.text(x = width+3, 
#                          y = p.get_y()+(height/2),
#                          s = "{:.0f}".format(width), 
#                          va = "center")
# =============================================================================
                   st.pyplot(fig)   
                  
                   def Cat_Model():
                            import joblib
                            pred_model = joblib.load('classifier_SACat.pkl.pkl')
                            return pred_model
                        
                   #clean_cat=CategoriseSA(data_processed)
      
                   pred_model=Cat_Model() 
                   
                   #categorise=pred_model.predict(clean_cat.statuses_without_stopwords)
                   
                   st.write(Cat_Model())  
#                    categorise=categorise.tolist()
#                    
#                    df_class=pd.DataFrame(categorise,columns=["Class_Label"])
#                    df_class=df_class.reset_index(drop=True)
#                    
#                    import numpy as np
#                    
#                    df_class['Tweet_Category'] = np.where((df_class['Class_Label'] ==0), 'Global Tweet', 'S.A Tweet')
#        
#                    df_cat=pd.concat([clean_cat,df_class],axis=1)
#                    
#                    
# =============================================================================
                   
# =============================================================================
#                    st.write('**SA vs Global Bar Graph**')
#                    
#                   
#                    ax = sns.countplot(y="Tweet_Category", data=df_cat)
# 
#                    for p in ax.patches:
#                              height = p.get_height() 
#                              width = p.get_width() 
#                              ax.text(x = width+3, 
#                              y = p.get_y()+(height/2),
#                              s = "{:.0f}".format(width), 
#                              va = "center")
#                    st.pyplot()
# =============================================================================
                      
# =============================================================================
#                 my_expander_Text = st.beta_expander("Show Text Analytics", expanded=True)
#                 with my_expander_Text:
#                 #if st.checkbox('Show Text Analytics'):   
#                   from wordcloud import WordCloud
#                   import matplotlib.pyplot as plt
#                   
#                   st.subheader("**Tweets WordCloud**")
#                   data_processed_text=clean_text(data_processed.statuses_text)
#   
#                   st.set_option('deprecation.showPyplotGlobalUse',False)  
#                   text=" ".join(clean_text for clean_text in data_processed_text.clean_text)
#                   wordcloud = WordCloud(max_words=100).generate(text)
# 
#                   # Display the generated image:
#                   plt.imshow(wordcloud, interpolation='bilinear')
#                   plt.axis("off")
#                   plt.title('Prevalent words in Tweets')
#                   plt.show()
#                   st.pyplot() 
#                   
#                   from PIL import Image
#                   st.subheader('Visual common words in each topic')
#                   st.write('The below visual shows the top salinent words in each topic. To have the interactive plot please click on link below visual')
#                   t_num=st.slider('Show most common words in each topic',max_value=3)
#                   if t_num==1:
#                       image = Image.open('Topic 2.jpg')
#                       st.image(image, caption='Topic 1 most used words')
#                   elif t_num==2:
#                       image = Image.open('Topic3.jpg')
#                       st.image(image, caption='Topic 2 most used words')
#                   elif t_num==3:
#                       image = Image.open('Topic1.jpg')
#                       st.image(image, caption='Topic 3 most used words')
#                   else:
#                      image = Image.open('Topic0.jpg')
#                      st.image(image, caption='Topic 3 most used words')
#             
# 
#                   url = 'https://htmlpreview.github.io/?https://github.com/YolandaNkalashe25/MIT805-Exam/blob/main/output_lda.html'
#                   #vis=loads(vis)
#                   import webbrowser
#                   if st.button('Open browser'):
#                     webbrowser.open_new_tab(url)
# 
#           
# # =============================================================================
# #                   import os
# #                   import base64
# #                   def get_binary_file_downloader_html(bin_file, file_label='File'):
# #                     with open(bin_file, 'rb') as f:
# #                      data = f.read()
# #                      bin_str = base64.b64encode(data).decode()
# #                      href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
# #                      return href
# #                  
# #     
# #                   st.markdown(get_binary_file_downloader_html(vis, 'Interactive Topic saliency plot'), unsafe_allow_html=True)
# #                   #cat_check=st.checkbox("Generate SA vs Global Bar Graph",value = False)
# #                   #if cat_check:
# # =============================================================================
#                   
#                 my_expander_Text_inter = st.beta_expander("Interactive Text Analytics", expanded=True)
#                 with my_expander_Text_inter:  
#                 #if st.checkbox('nteractive Text Analytics'): 
#                     
#                   st.subheader('Interactive Unsupervised Learning on Microblogs')
#                   import emoji
#                   from sklearn.cluster import KMeans
#                   from sklearn.cluster import MiniBatchKMeans
#                   from sklearn.decomposition import PCA
#                   from sklearn.manifold import TSNE
#                   
#                   def cluster(Document, n_clusters):
#                       vectorizer_tfidf = TfidfVectorizer(stop_words='english')
#                       vectorizer_tfidf.fit(Document)
#                       X_tfidf = vectorizer_tfidf.transform(Document)
#     
#                       tfidf_feature_names = vectorizer_tfidf.get_feature_names()
#     
#                       pca=PCA(n_components=2)
#                       X_pca=pca.fit_transform(X_tfidf.toarray())
#     
#                       clf=MiniBatchKMeans(n_clusters=n_clusters,compute_labels=True)
#                       clf.fit(X_tfidf)
#                       within_cluster=clf.inertia_
#     
#                       cluster_labels = clf.predict(X_tfidf)
#     
#                       return X_pca, cluster_labels
#                       
#                   num_clusters=st.slider('Number of unsupervised clusters',min_value=3, max_value=10,step=1)
#                   Document=data_processed.statuses_text
#                   X_pca, cluster_labels=cluster(Document,num_clusters)
#             
#                   import matplotlib.pyplot as plt
#                   import numpy as np
#                   
#                   fig = plt.figure()
#                   ax=fig.add_subplot(projection='3d')
# 
#                   # Plot scatterplot data (20 2D points per colour) on the x and z axes.
#                   colors = ('r', 'g', 'b', 'k')
# 
#                   # Fixing random state for reproducibility
#                   np.random.seed(19680801)
# 
#                   x = X_pca[:,0]
#                   y = X_pca[:,1]
#                   
#                   scatter=ax.scatter(x, y, zs=0, zdir='y', c=cluster_labels, label='points in (x, z)')
#                   
#                   ax.legend()
#                   ax.set_xlabel('X')
#                   ax.set_ylabel('Y')
#                   ax.set_zlabel('Z')
#                   legend1 = ax.legend(*scatter.legend_elements(),
#                      loc="lower left")
#                   ax.add_artist(legend1)
#                   
#                   st.write(fig)
# =============================================================================
              
if __name__=='__main__':
          main()   
