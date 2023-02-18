import pandas as pd
import math

from nltk import pos_tag

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')



def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_text =pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def preprocessing(d):
# 1-  Map Textual labels to numeric using Label Encoder  &&   chek if there any null values

    data = pd.DataFrame()
    data["text"] = d["text"]
    data["label_encode"] = LabelEncoder().fit_transform(d["category"])
    data["label_encode"]=data["label_encode"].astype(str)
    data["label"]=d["category"]

# 2- convert text to lower case
    data["text"] = data["text"].map(lambda x: x.lower())
# 3- remove 's
    data["text"] = data["text"].str.replace("'s", "")
   
#  4- remove stop words

    # define list of the stop words that we want use
    stop_words=set(stopwords.words("english"))
    
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
#   


#   5-   remove punctuations 
#    a) Remove punctuation from each token.

    data['text'] = data['text'].apply(remove_punctuations)

#   b)   Filter out remaining tokens that are not alphabetic.
    
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x.isalpha()))
    
#   6-   Lemmatization     
#    ******* Lemmatization with pos tags because the lemmatization process depends on the POS tag to come up with the correct lemma.   **********

    data["text"] = data["text"].apply(lambda text: lemmatize_words(text))
    return data
    
def tfidf_tf(data):
    val=[]
    l=data.columns
    for c in l:
        
        val=data[c].tolist()
        l= len (val)
        for i , item in enumerate(val):
            if val[i] != 0.0:
                val[i]=1
            else:
                val[i] =0
        data[c] = val
    return data 


def collect_features(df_train):
    s00=[]
    s11=[]
    s22=[]
    l=len(df_train.columns)
    for index, row in df_train.iterrows():
      if df_train.get_value(index,'labels')=='0':
          s0=[]
          for c in df_train.columns[0:l-1]:
               if df_train.get_value(index,c)!=0:
                      s0.append(str(c))
          s00.append(s0)
      elif df_train.get_value(index,'labels')=='1':
          s1=[]
          for c in df_train.columns[0:l-1]:
               if df_train.get_value(index,c)!=0:
                      s1.append(str(c))
          s11.append(s1)
      else:
          s2=[]
          for c in df_train.columns[0:l-1]:
               if df_train.get_value(index,c)!=0:
                      s2.append(c)
          s22.append(s2)
                     
    return  s00,s11,s22

def compute_allconf(firstCandidateSet ,frequentitemsets_without_1st):
        item=[]
        support=[]
        for index, row in frequentitemsets_without_1st.iterrows():
    
            temp=frequentitemsets_without_1st.get_value(index,'itemsets')
            
            supp_temp=frequentitemsets_without_1st.get_value(index,'support')
            for i in temp:
               for j in range(len(firstCandidateSet)):
                   if j%2==0:
                       if i==firstCandidateSet[j]:
                          item.append(i)
                          support.append(firstCandidateSet[j+1])
                          
            max_itemset=max(support)
            frequentitemsets_without_1st.at[index,'all_conf']=supp_temp/max_itemset
        print("=======================================================")
        print frequentitemsets_without_1st
        return frequentitemsets_without_1st
    
def prune_itemset(all_cof_threshold,frequentitemsets_without_1st):
#    1- prune itemsets that have all_conf <= threshold
    indexes=frequentitemsets_without_1st[frequentitemsets_without_1st['all_conf']<=all_cof_threshold].index
    frequent_itemsets_conf=pd.DataFrame(frequentitemsets_without_1st.drop(indexes,inplace=False))
   
     
#    2-  prune itemset if it is supset of general itemset and its allconf <= allconf of general itemset
    ii=[]
    
#    store itemsets and thier support in dict
    all_itemsets=frequent_itemsets_conf['itemsets'].tolist()
    all_itemsets_all_conf=frequent_itemsets_conf['all_conf'].tolist()
    all_itemset_dic=dict(zip(map(float,all_itemsets_all_conf),all_itemsets))
    
    for index, row in frequent_itemsets_conf.iterrows():
        item1=frequent_itemsets_conf.get_value(index,'itemsets')
        allconf_item1=frequent_itemsets_conf.get_value(index,'all_conf')
        flag=False
        for allconf,itemsset in all_itemset_dic.items():
            if (len (item1)<len( itemsset) and set(item1).issubset(set(itemsset))):
                allconf_itemsset=allconf
                if(allconf_item1<=allconf_itemsset):
                    flag=True
                    break
        if  flag:
            ii.append(index)
       
    frequent_itemsets_pruned=pd.DataFrame(frequent_itemsets_conf.drop(ii,inplace=False))       
    return     frequent_itemsets_pruned 

def get_features(filtered_itemsets): 
    #count term frequency
    c= Counter(x for xs in filtered_itemsets for x in set(xs))
    no_of_items=len(c) 
    
    for key in c:
        c[key]=float(c[key])/no_of_items
    #   select term that more than threshold
    for k,v in c.items():
        if v<=0.048:
            del c[k]
    
    freq_items=list(c.keys())
    
    return freq_items
                     
def fs_ass(c) :
# 1- convert each list to transaction 0,1(0:not found,1:found)
    te = TransactionEncoder()
    te_ary = te.fit(c).transform(c)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
#    generate all 1st_itemset
    itemsets_1 = apriori(df,min_support=0.0,max_len=1, use_colnames=True)

#    store item set and its support in list
    firstCandidateSet=[]
    for index, row in itemsets_1.iterrows():
        firstCandidateSet.append(''.join(itemsets_1.get_value(index,'itemsets')))
        firstCandidateSet.append(itemsets_1.get_value(index,'support')) 
    
#    generate frequent_itemsets
    frequent_itemsets = apriori(df, min_support=0.045, use_colnames=True)
    
#    add length and index col to frequent_itemsets dataframe
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets['index']=frequent_itemsets.index
   
   
    # drop 1st_itemset from frequent_itemsets
    indexes_1st=frequent_itemsets[frequent_itemsets['length']==1].index
    frequentitemsets_without_1st=pd.DataFrame(frequent_itemsets.drop(indexes_1st,inplace=False))
    print frequentitemsets_without_1st
#    compute all confedence for each item set
    frequentitemsets_without_1st=compute_allconf(firstCandidateSet ,frequentitemsets_without_1st)

# pruning item sets
    all_cof_threshold=0.13
    frequent_itemsets_pruned=prune_itemset(all_cof_threshold,frequentitemsets_without_1st)
    # get unique items from  pruned frequent itemsets
    filtered_itemsets=frequent_itemsets_pruned['itemsets'].tolist()
#    unique_filtered_itemsets=list(set(chain(*filtered_itemsets)))
    
    features=get_features(filtered_itemsets)
    
    
    return  features

    
    
def fit_features(df_train,features_S0,features_S1,features_S2):
    
    l=len(df_train.columns)
    for index, row in df_train.iterrows():
      if df_train.get_value(index,'labels')=='0':
               for c in df_train.columns[0:l-1]:
                    if df_train.get_value(index,c)!=0:
                        if c not in features_S0:
                             df_train.at[index,c]=0
                        else:
                             df_train.at[index,c]=1
      elif df_train.get_value(index,'labels')=='1':
         for c in df_train.columns[0:l-1]:
                    if df_train.get_value(index,c)!=0:
                        if c not in features_S1:
                             df_train.at[index,c]=0
                        else:
                             df_train.at[index,c]=1 
      else:
          for c in df_train.columns[0:l-1]:
                    if df_train.get_value(index,c)!=0:
                        if c not in features_S2:
                             df_train.at[index,c]=0
                        else:
                             df_train.at[index,c]=1
    return df_train

       
        

def main():    
    # 1- load data set 
    
    df_bbc = pd.read_csv('D://flash 2//phd thesis//papers_exper//paper 3//datasets//Text-Intent-Classification-master//dataset.csv',encoding="utf-8")
    df_bbc=df_bbc.astype(str)
#    2- preprocessing
    data= preprocessing(df_bbc)
    print data.head()
    
#   3- split data    
    X_train, X_test, y_train, y_test = train_test_split(data['text'],data['label'], test_size=0.30,random_state=8)

#  4- tf_idf   
    ngram_range = (1,1)
    min_df = 10
    max_df = 85
    tfidf = TfidfVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            lowercase=False,
                            max_df=max_df,
                            min_df=min_df,
                            norm='l2',
                            sublinear_tf=True)
                            
    features_train = tfidf.fit_transform(X_train).toarray()
    
    labels_train = y_train
    labels_train.reset_index(drop=True,inplace=True)
    df_train = pd.DataFrame(features_train, columns = tfidf.get_feature_names())
    df_train['labels']=labels_train
    
    
    
    features_test = tfidf.transform(X_test).toarray()
    
    labels_test = y_test
    labels_test.reset_index(drop=True,inplace=True)
    df_test = pd.DataFrame(features_test, columns = tfidf.get_feature_names())
    df_test['labels']=labels_test
    
    
   #  5-  collect features from train dataframe in each class && convert it to transactions
    s00,s11,s22=collect_features(df_train)
#   business,tech,sport,entertainment,politics=collect_features(df_train)
   
    
    #  6- feature selection
    
    print ("total number of extracted features are :  ")

    features_S0=fs_ass(s00)
    print (features_S0)
    
    features_S1=fs_ass(s11)
    print (features_S1)
    
    features_S2=fs_ass(s22)
    print (features_S2)
   
    
    total_features=len(features_S0)+len(features_S1)+len(features_S2)
    print (total_features)
    
#
#
#
    #   7-    fit features to train and test datasets  
    df_train_copy=df_train.copy()  
    df_train= fit_features(df_train,features_S0,features_S1,features_S2)
    df_test =fit_features(df_test ,features_S0,features_S1,features_S2)
    
                             
    
    
    
    l_train=len (df_train.columns)
    l_test=len (df_test.columns)
   
    df_train_X=df_train[df_train.columns[:l_train-1]]
    df_train_Y=df_train['labels']

    df_test_X=df_test[df_test.columns[:l_test-1]]
    df_test_Y=df_test['labels']
    
    
    
    
    #   7-    apply classifiers and evaluate their performance  

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    from sklearn.metrics import classification_report
    
    ACC=[]
    F=[]
    classifiers=['Random Forest','Logistic Regression','Decision Tree','GaussianNB']
    models=[RandomForestClassifier(random_state=1),LogisticRegression(random_state=1),DecisionTreeClassifier(random_state=1),GaussianNB()]   
         
    for i in models:
        model = i
        model.fit(df_train_X, df_train_Y)
        prediction=model.predict(df_test_X)
#        print "Classification report for %s: " %(i)
#        print(classification_report(prediction,df_test_Y))
        ACC.append(metrics.accuracy_score(prediction,df_test_Y)*100)
        F.append(math.ceil(metrics.f1_score(prediction,df_test_Y,average='micro')*100))
    metrics={'Accuracy':ACC,'F-measure':F}    
    new_models_dataframe=pd.DataFrame(metrics,index=classifiers)   
    
    
    print(new_models_dataframe)       

    
    




    
    
main()

