import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import os

plt.style.use('default')
sns.set_palette(["#440D0D", "#12652E", "#12115D", "#195C0C"])

class AnalizaEksploruese:
    def __init__(self):
        self.ngjyrat = ["#4E3030", "#055791", "#301F58", "#2E4B3D"]
    
    def ngarko_te_dhenat(self):
        te_dhenat = joblib.load('models/processed_data.joblib')
        df = pd.DataFrame({
            'text': list(te_dhenat['X_train']) + list(te_dhenat['X_test']),
            'label': list(te_dhenat['y_train']) + list(te_dhenat['y_test'])
        })
        df['category'] = [te_dhenat['target_names'][etiketa] for etiketa in df['label']]
        return df, te_dhenat['target_names']
    
    def vizualizo_shperndarjen_e_klaseve(self, df):
        plt.figure(figsize=(10, 6))
        ax = df['category'].value_counts().plot(kind='bar', color=self.ngjyrat)
        plt.title('Shpërndarja e Kategorive të Lajmeve', fontsize=16, fontweight='bold')
        plt.xlabel('Kategoritë', fontsize=12)
        plt.ylabel('Numri i Dokumenteve', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for i, v in enumerate(df['category'].value_counts()):
            ax.text(i, v + 10, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('shperndarja_e_klaseve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def vizualizo_fjalet_kryesore(self, df, numri_i_fjaleve=20):
        vektorizues = CountVectorizer(max_features=numri_i_fjaleve, stop_words='english')
        X = vektorizues.fit_transform(df['text'])
        fjalet = vektorizues.get_feature_names_out()
        numërimet = X.sum(axis=0).A1
        frekuencat = dict(zip(fjalet, numërimet))
        
        fjalet_renditur = sorted(frekuencat.items(), key=lambda x: x[1], reverse=True)[:numri_i_fjaleve]
        
        plt.figure(figsize=(12, 8))
        fjalet, numërimet = zip(*fjalet_renditur)
        shiritat = plt.barh(fjalet, numërimet, color=self.ngjyrat[0])
        plt.title(f'{numri_i_fjaleve} Fjalët Më të Frekuentuara', fontsize=16, fontweight='bold')
        plt.xlabel('Frekuenca', fontsize=12)
        plt.gca().invert_yaxis()
        
        for shirit in shiritat:
            gjeresia = shirit.get_width()
            plt.text(gjeresia + 10, shirit.get_y() + shirit.get_height()/2, 
                    f'{int(gjeresia)}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('fjalet_kryesore.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analizo_gjatesine_e_tekstit(self, df):
        df['gjatesia_tekstit'] = df['text'].str.len()
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='category', y='gjatesia_tekstit', palette=self.ngjyrat)
        plt.title('Shpërndarja e Gjatësisë së Tekstit sipas Kategorisë', fontsize=16, fontweight='bold')
        plt.xlabel('Kategoria', fontsize=12)
        plt.ylabel('Gjatësia e Tekstit (karaktere)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('shperndarja_e_gjatesise_se_tekstit.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def shfaq_tekste_shembull(self, df):
        print("\nTekste shembull nga çdo kategori:")
        for kategori in df['category'].unique():
            print(f"\n--- {kategori} ---")
            teksti_shembull = df[df['category'] == kategori].iloc[0]['text']
            print(teksti_shembull[:200] + "..." if len(teksti_shembull) > 200 else teksti_shembull)

def main():
    analizuesi = AnalizaEksploruese()
    
    df, emrat_kategoriye = analizuesi.ngarko_te_dhenat()
    
    print("Përmbledhje e Dataset-it:")
    print(f"Totali i dokumenteve: {len(df)}")
    print(f"Kategoritë: {emrat_kategoriye}")
    print("\nShpërndarja e klasave:")
    print(df['category'].value_counts())
    
    analizuesi.vizualizo_shperndarjen_e_klaseve(df)
    analizuesi.vizualizo_fjalet_kryesore(df)
    analizuesi.analizo_gjatesine_e_tekstit(df)
    analizuesi.shfaq_tekste_shembull(df)
    
    print("\nVizualizimet u ruajtën si skedarë PNG!")

if __name__ == "__main__":
    main()