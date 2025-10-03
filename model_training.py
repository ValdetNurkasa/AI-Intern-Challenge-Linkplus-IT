import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Bonus 2: Compare at least...
from sklearn.model_selection import cross_val_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.modelet = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        self.rezultatet = {}
    
    def ngarko_te_dhenat(self):
        te_dhenat = joblib.load('models/processed_data.joblib')
        return te_dhenat['X_train'], te_dhenat['X_test'], te_dhenat['y_train'], te_dhenat['y_test'], te_dhenat['target_names']
    
    #Bonus 1: TF-IDF
    def krijo_vektor_tfidf(self, X_train, X_test, max_features=5000):
        vektorizues = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        X_train_tfidf = vektorizues.fit_transform(X_train)
        X_test_tfidf = vektorizues.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf, vektorizues
    
    def trajno_modelet(self, X_train, X_test, y_train, y_test):
        X_train_tfidf, X_test_tfidf, vektorizues = self.krijo_vektor_tfidf(X_train, X_test)
        
        joblib.dump(vektorizues, 'models/tfidf_vectorizer.joblib')
        
        print("Duke trajnuar modelet...")
        for emri, model in self.modelet.items():
            print(f"\nDuke trajnuar {emri}...")
            
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            saktësia = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
            
            self.rezultatet[emri] = {
                'model': model,
                'accuracy': saktësia,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"{emri} - Saktësia: {saktësia:.4f}, CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            joblib.dump(model, f'models/{emri.lower().replace(" ", "_")}.joblib')
    
    def vizualizo_matricen_e_konfuzionit(self, y_true, y_pred, emrat_kategoriye, emri_modelit):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emrat_kategoriye, yticklabels=emrat_kategoriye)
        plt.title(f'Matrica e Konfuzionit - {emri_modelit}', fontsize=16, fontweight='bold')
        plt.xlabel('Etiketa e Parashikuar', fontsize=12)
        plt.ylabel('Etiketa e Vërtetë', fontsize=12)

        #Bonus 3:...
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{emri_modelit.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def krahaso_modelet(self):
        modelet = list(self.rezultatet.keys())
        saktësitë = [self.rezultatet[model]['accuracy'] for model in modelet]
        cv_mesataret = [self.rezultatet[model]['cv_mean'] for model in modelet]
        cv_devijimet = [self.rezultatet[model]['cv_std'] for model in modelet]
        
        x = np.arange(len(modelet))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, saktësitë, width, label='Saktësia e Testimit', color='#4ECDC4')
        bars2 = ax.bar(x + width/2, cv_mesataret, width, label='CV Mesatare', yerr=cv_devijimet, 
                      capsize=5, color='#FF6B6B')
        
        #Bonus 2:...
        ax.set_xlabel('Modelet', fontsize=12)
        ax.set_ylabel('Saktësia', fontsize=12)
        ax.set_title('Krahasimi i Modeleve', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modelet, rotation=45, ha='right')
        ax.legend()
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('krahasimi_i_modeleve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def printo_raportin_e_detajuar(self, y_test, emrat_kategoriye):
        for emri in self.rezultatet.keys():
            print(f"\n{'='*50}")
            print(f"Raport i Detajuar - {emri}")
            print(f"{'='*50}")
            y_pred = self.rezultatet[emri]['predictions']
            print(classification_report(y_test, y_pred, target_names=emrat_kategoriye))

def main():
    trajneri = ModelTrainer()
    
    X_train, X_test, y_train, y_test, emrat_kategoriye = trajneri.ngarko_te_dhenat()
    
    print("Të dhënat u ngarkuan me sukses!")
    print(f"Mostrat e trajnimit: {len(X_train)}")
    print(f"Mostrat e testimit: {len(X_test)}")
    print(f"Kategoritë: {emrat_kategoriye}")
    
    trajneri.trajno_modelet(X_train, X_test, y_train, y_test)
    trajneri.krahaso_modelet()
    trajneri.printo_raportin_e_detajuar(y_test, emrat_kategoriye)
    
    for emri_modelit in trajneri.rezultatet.keys():
        y_pred = trajneri.rezultatet[emri_modelit]['predictions']
        trajneri.vizualizo_matricen_e_konfuzionit(y_test, y_pred, emrat_kategoriye, emri_modelit)
    
    modeli_më_i_mirë = max(trajneri.rezultatet.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n{'='*60}")
    print(f"MODELI MË I MIRË: {modeli_më_i_mirë[0]}")
    print(f"Saktësia: {modeli_më_i_mirë[1]['accuracy']:.4f}")
    print(f"Cross-validation Score: {modeli_më_i_mirë[1]['cv_mean']:.4f} (+/- {modeli_më_i_mirë[1]['cv_std'] * 2:.4f})")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()