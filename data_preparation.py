import pandas as pd
import numpy as np
import re
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs('models', exist_ok=True)

print("Duke shkarkuar të dhënat NLTK...")
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessor:
    def __init__(self):
        self.fjalet_stop = set(stopwords.words('english'))
        
    def pastro_tekstin(self, tekst):
        if not isinstance(tekst, str):
            return ""
            
        tekst = tekst.lower()
        tekst = re.sub(r'[^a-zA-Z\s]', '', tekst)
        tekst = ' '.join(tekst.split())
        
        fjalet = word_tokenize(tekst)
        fjalet = [fjale for fjale in fjalet if fjale not in self.fjalet_stop and len(fjale) > 2]
        
        return ' '.join(fjalet)
    
    def ngarko_te_dhenat(self):
        print("Duke ngarkuar dataset-in 20 Newsgroups...")
        
        kategorite = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.misc']
        
        newsgroups_train = fetch_20newsgroups(
            subset='train',
            categories=kategorite,
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=42
        )
        
        newsgroups_test = fetch_20newsgroups(
            subset='test',
            categories=kategorite,
            remove=('headers', 'footers', 'quotes'),
            shuffle=True,
            random_state=42
        )
        
        tekstet = list(newsgroups_train.data) + list(newsgroups_test.data)
        etiketat = list(newsgroups_train.target) + list(newsgroups_test.target)
        
        df = pd.DataFrame({
            'text': tekstet,
            'label': etiketat,
            'category': [newsgroups_train.target_names[etiketa] for etiketa in etiketat]
        })
        
        return df, newsgroups_train.target_names
    
    def perpun_te_dhenat(self, df):
        print("Duke pastruar të dhënat e tekstit...")
        df['cleaned_text'] = df['text'].apply(self.pastro_tekstin)
        df = df[df['cleaned_text'].str.len() > 0]
        return df

def main():
    perpunsuesi = DataPreprocessor()
    
    df, emrat_kategoriye = perpunsuesi.ngarko_te_dhenat()
    print(f"Madhësia origjinale e dataset-it: {df.shape}")
    print(f"Kategoritë: {emrat_kategoriye}")
    
    df = perpunsuesi.perpun_te_dhenat(df)
    print(f"Madhësia e dataset-it pas pastrimit: {df.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], 
        test_size=0.2, random_state=42, stratify=df['label']
    )
    
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'target_names': emrat_kategoriye
    }, 'models/processed_data.joblib')
    
    print("Përpunimi i të dhënave u përfundua!")
    print(f"Mostrat e trajnimit: {len(X_train)}")
    print(f"Mostrat e testimit: {len(X_test)}")

if __name__ == "__main__":
    main()