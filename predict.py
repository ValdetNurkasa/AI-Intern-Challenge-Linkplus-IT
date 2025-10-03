import joblib
import numpy as np
import pandas as pd
from data_preparation import DataPreprocessor

class KlasifikuesiITekstit:
    def __init__(self, model_path='models/logistic_regression.joblib', 
                 vectorizer_path='models/tfidf_vectorizer.joblib',
                 target_names_path='models/processed_data.joblib'):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.perpunsuesi = DataPreprocessor()
            
            te_dhenat = joblib.load(target_names_path)
            self.emrat_kategoriye = te_dhenat['target_names']
            print(f"Modeli u ngarkua me sukses! Kategoritë: {self.emrat_kategoriye}")
            
        except Exception as e:
            print(f"Gabim gjatë ngarkimit të modeleve: {e}")
            print("Ju lutem sigurohuni që keni ekzekutuar data_preparation.py dhe model_training.py fillimisht!")
            raise e
    
    def parashiko(self, tekst):
        teksti_i_pastro = self.perpunsuesi.pastro_tekstin(tekst)
        
        if len(teksti_i_pastro) == 0:
            return {
                'kategoria_e_parashikuar': 'E panjohur',
                'pikët_besimi': {emri: 0.0 for emri in self.emrat_kategoriye},
                'teksti_i_pastro': teksti_i_pastro,
                'gabim': 'Teksti shumë i shkurtër pas pastrimit'
            }
        
        teksti_tfidf = self.vectorizer.transform([teksti_i_pastro])
        parashikimi = self.model.predict(teksti_tfidf)[0]
        probabilitetet = self.model.predict_proba(teksti_tfidf)[0]
        
        pikët_besimi = {
            self.emrat_kategoriye[i]: float(probabilitetet[i]) 
            for i in range(len(self.emrat_kategoriye))
        }
        
        return {
            'kategoria_e_parashikuar': self.emrat_kategoriye[parashikimi],
            'pikët_besimi': pikët_besimi,
            'teksti_i_pastro': teksti_i_pastro
        }

def main():
    try:
        klasifikuesi = KlasifikuesiITekstit()
    except Exception as e:
        print("Dështoi inicializimi i klasifikuesit. Ju lutem ekzekutoni hapat e trajnimit fillimisht.")
        return
    
    print("=" * 60)
    print("Demonstrim i Klasifikimit të Tekstit")
    print("=" * 60)
    print("Kategoritë:", klasifikuesi.emrat_kategoriye)
    print("\nShkruani tekst për të klasifikuar (shkruani 'dil' për të dalë):")
    
    shembuj_test = [
        "kompjuter grafika dhe teknika 3D modelimi",
        "lojtarët e bejzbolit dhe mesataret e tyre të goditjeve",
        "kërkime mjekësore dhe prova klinike",
        "debate politike dhe politika qeveritare"
    ]
    
    print("\nShembuj test që mund të provoni:")
    for i, shembull in enumerate(shembuj_test, 1):
        print(f"{i}. {shembull}")
    
    while True:
        try:
            input_i_perdoruesit = input("\nShkruani tekst: ").strip()
            
            if input_i_perdoruesit.lower() == 'dil':
                break
                
            if not input_i_perdoruesit:
                print("Ju lutem shkruani tekst.")
                continue
            
            rezultati = klasifikuesi.parashiko(input_i_perdoruesit)
            
            print(f"\nKategoria e parashikuar: {rezultati['kategoria_e_parashikuar']}")
            print("\nPikët e besimit:")
            for kategori, pike in sorted(rezultati['pikët_besimi'].items(), 
                                        key=lambda x: x[1], reverse=True):
                print(f"  {kategori}: {pike:.4f}")
                
            if 'gabim' in rezultati:
                print(f"Shënim: {rezultati['gabim']}")
                
        except KeyboardInterrupt:
            print("\n\nDuke dalë...")
            break
        except Exception as e:
            print(f"Gabim gjatë parashikimit: {e}")

if __name__ == "__main__":
    main()