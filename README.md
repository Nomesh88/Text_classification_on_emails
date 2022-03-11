# Text_classification_on_emails

# Importing required packages
      import numpy as np 
      import pandas as pd
      import matplotlib.pyplot as plt
      import os
      #---------------------------------------Text Processing------------------------------------------------------------#
      from sklearn.feature_extraction.text import TfidfVectorizer
      from string import punctuation
      #------------------------------------Metrics and Validation---------------------------------------------------------#
      from sklearn.model_selection import train_test_split, RandomizedSearchCV
      from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
      #-------------------------------------Models to be trained----------------------------------------------------------#
      from sklearn.ensemble import StackingClassifier, VotingClassifier
      from sklearn.linear_model import LogisticRegression, SGDClassifier
      from sklearn.naive_bayes import MultinomialNB
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.cluster import KMeans

  # Reading the categories of different files
        names = []
        base = 'C:/Users/nomesh.palakaluri.EMEA/OneDrive - Drilling Info/Desktop/Text model/Data/'
        with os.scandir(base) as entries:
            for entry in entries:
                if(entry.is_file() == False):
                    names.append(entry.name)
        print(names)
  
  # 
      files = {}
      unique = []
      for name in names:
          path = base + name+'/'
          x = []
          with os.scandir(path) as entries:
              for entry in entries:
                  if(entry.is_file()):
                      x.append(entry.name)
          files[name] = x
          files[name].sort()

          for i in range(len(names)):
        x = files[names[i]]
        for j in x:
            for k in range(i+1, len(names)):
                key = names[k]
                if j in files[key]:
                    files[key].remove(j)
# splitting the text files into sentences and classes
      data = {}
      i = 0

      for genre in files.keys() :
          texts = files[genre]
          for text in texts:
              if text in files[genre]:
                  path = base + genre + '/' + text
                  with open(path, "r", encoding = "latin1") as file:
                      data[i] = file.readlines()
                      i = i+1
                  data[i-1] = [" ".join(data[i-1]), genre] 

      data = pd.DataFrame(data).T
      print(data.shape)
      data.columns = ['Text', 'Class']
#
  unique = list(data.Text.unique())
  len(unique)

# Converting the updated data to a dictionary
  dic = dict(data)

## unique values in the documents
  uni = {}
  i = 0
  for k in range(len(list(dic['Text']))):
      if dic['Text'][k] in unique:
          uni[i] = [dic['Text'][k], dic['Class'][k]]
          unique.remove(dic['Text'][k])
          i += 1
# classification into text and type of class
    data = pd.DataFrame(uni).T
    print(data.shape)
    data.columns = ['Text', 'Class']
# cleaning text
    import nltk.corpus
    import regex
    nltk.download('stopwords')
    from nltk.corpus import stopwords 
    from nltk.tokenize import WordPunctTokenizer
    from string import punctuation
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    stop = stopwords.words('english')

    for punct in punctuation:
        stop.append(punct)

    def filter_text(text, stop_words):
        word_tokens = WordPunctTokenizer().tokenize(text.lower())
        filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha() and len(w) > 3]
        filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 
        return " ".join(filtered_text)
# filtering the text
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    data["filtered_text"] = data.Text.apply(lambda x : filter_text(x, stop)) 
    data.head()
# top 10 words to claasify the text into crime class
        all_text = " ".join(data[data.Class == "Crime"].filtered_text) 
        count = pd.DataFrame(all_text.split(), columns = ['words'])
        top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]
        print(top_10)
 # top 10 words to claasify the text into politics class
        all_text = " ".join(data[data.Class == "Politics"].filtered_text)
        count = pd.DataFrame(all_text.split(), columns = ['words'])
        top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]
        print(top_10)
## top 10 words to claasify the text into science class
  all_text = " ".join(data[data.Class == "Science"].filtered_text)
  count = pd.DataFrame(all_text.split(), columns = ['words'])
  top_10 = count[count['words'].isin(list(count.words.value_counts()[:10].index[:10]))]
  print(top_10)
# #vectorizing the data 
        tfidf = TfidfVectorizer(lowercase=False)
        train_vec = tfidf.fit_transform(data['filtered_text'])
        train_vec.shape
# #classifying the classes and assigning some values
      data['classification'] = data['Class'].replace(['Crime','Politics','Science'],[0,1,2])
# splitting and train of the data
      x_train, x_val, y_train, y_val = train_test_split(train_vec,data['classification'], stratify=data['classification'], test_size=0.2)
# model
      C = np.arange(0, 1, 0.001)
      max_iter = range(100, 500)
      warm_start = [True, False]
      solver = ['lbfgs', 'newton-cg', 'liblinear']
      penalty = ['l2', 'l1']

      params = {
          'C' : C,
          'max_iter' : max_iter,
          'warm_start' : warm_start,
          'solver' : solver,
          'penalty' : penalty
      }

      random_search = RandomizedSearchCV(
          estimator = LogisticRegression(random_state = 1),
          param_distributions = params,
          n_iter = 100,
          cv = 3,
          n_jobs = -1,
          random_state = 1,
          verbose = 1
      ).fit(x_train, y_train)

      random_search.best_params_
 # accuracy
      model_lr = random_search.best_estimator_
      model_lr.score(x_train, y_train)
 # predictions
        predicted = model_lr.predict(x_val)

        lr_acc = accuracy_score(y_val,predicted)
        lr_cop = cohen_kappa_score(y_val,predicted)
        lr = pd.DataFrame([lr_acc, lr_cop], columns = ['Logistic Regression with RandomizedSearchCV'])

        print("Test score: {:.2f}".format(lr_acc))



