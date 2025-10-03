#Bonus 4:...
from flask import Flask, request, jsonify
import joblib
from data_preparation import DataPreprocessor
import numpy as np

app = Flask(__name__)

klasifikuesi = None
vektorizuesi = None
emrat_kategoriye = None
perpunsuesi = DataPreprocessor()

def ngarko_modelet():
    global klasifikuesi, vektorizuesi, emrat_kategoriye
    
    try:
        klasifikuesi = joblib.load('models/logistic_regression.joblib')
        vektorizuesi = joblib.load('models/tfidf_vectorizer.joblib')
        
        te_dhenat = joblib.load('models/processed_data.joblib')
        emrat_kategoriye = te_dhenat['target_names']
        
        print("Modelet u ngarkuan me sukses!")
        print(f"Kategoritë e disponueshme: {emrat_kategoriye}")
        
    except Exception as e:
        print(f"Gabim gjatë ngarkimit të modeleve: {e}")
        print("Ju lutem ekzekutoni data_preparation.py dhe model_training.py fillimisht!")
        raise e

@app.route('/')
def faqja_kryesore():
    return jsonify({
        'mesazh': 'API për Klasifikimin e Tekstit',
        'version': '1.0',
        'endpointet': {
            '/predict': 'POST - Parashiko kategorinë e tekstit',
            '/batch_predict': 'POST - Parashiko kategori për tekste të shumta',
            '/categories': 'GET - Merr kategoritë e disponueshme',
            '/health': 'GET - Kontrollo gjendjen e API'
        }
    })

@app.route('/health', methods=['GET'])
def kontrollo_gjendjen():
    status = 'shëndetshëm' if klasifikuesi is not None else 'modelet nuk janë ngarkuar'
    return jsonify({
        'status': status, 
        'modelet_ngarkuar': klasifikuesi is not None,
        'kategoritë_ngarkuar': emrat_kategoriye is not None
    })

@app.route('/categories', methods=['GET'])
def merr_kategorite():
    if emrat_kategoriye is None:
        return jsonify({'gabim': 'Modelet nuk janë ngarkuar'}), 500
    return jsonify({'kategoritë': emrat_kategoriye})

@app.route('/predict', methods=['POST'])
def parashiko():
    try:
        if klasifikuesi is None:
            return jsonify({'gabim': 'Modelet nuk janë ngarkuar'}), 500
        
        te_dhenat = request.get_json()
        
        if not te_dhenat or 'teksti' not in te_dhenat:
            return jsonify({'gabim': 'Nuk u dha tekst'}), 400
        
        teksti = te_dhenat['teksti']
        
        if not isinstance(teksti, str):
            return jsonify({'gabim': 'Teksti duhet të jetë string'}), 400
        
        teksti_i_pastro = perpunsuesi.pastro_tekstin(teksti)
        
        if len(teksti_i_pastro) == 0:
            return jsonify({
                'gabim': 'Teksti shumë i shkurtër pas përpunimit',
                'teksti_i_pastro': teksti_i_pastro
            }), 400
        
        teksti_tfidf = vektorizuesi.transform([teksti_i_pastro])
        
        parashikimi = klasifikuesi.predict(teksti_tfidf)[0]
        probabilitetet = klasifikuesi.predict_proba(teksti_tfidf)[0]
        
        pikët_besimi = {
            emrat_kategoriye[i]: float(probabilitetet[i]) 
            for i in range(len(emrat_kategoriye))
        }
        
        pergjigja = {
            'kategoria_e_parashikuar': emrat_kategoriye[parashikimi],
            'pikët_besimi': pikët_besimi,
            'teksti_i_pastro': teksti_i_pastro,
            'teksti_origjinal': teksti
        }
        
        return jsonify(pergjigja)
        
    except Exception as e:
        return jsonify({'gabim': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def parashiko_shume():
    try:
        if klasifikuesi is None:
            return jsonify({'gabim': 'Modelet nuk janë ngarkuar'}), 500
        
        te_dhenat = request.get_json()
        
        if not te_dhenat or 'tekstet' not in te_dhenat:
            return jsonify({'gabim': 'Nuk u dhanë tekste'}), 400
        
        tekstet = te_dhenat['tekstet']
        
        if not isinstance(tekstet, list):
            return jsonify({'gabim': 'tekstet duhet të jenë listë'}), 400
        
        if len(tekstet) > 100:
            return jsonify({'gabim': 'Numri i teksteve është shumë i madh. Maksimumi 100 tekste.'}), 400
        
        tekstet_e_pastra = [perpunsuesi.pastro_tekstin(tekst) for tekst in tekstet]
        
        tekstet_tfidf = vektorizuesi.transform(tekstet_e_pastra)
        
        parashikimet = klasifikuesi.predict(tekstet_tfidf)
        probabilitetet = klasifikuesi.predict_proba(tekstet_tfidf)
        
        rezultatet = []
        for i, tekst in enumerate(tekstet):
            pikët_besimi = {
                emrat_kategoriye[j]: float(probabilitetet[i][j]) 
                for j in range(len(emrat_kategoriye))
            }
            
            rezultatet.append({
                'teksti_origjinal': tekst,
                'kategoria_e_parashikuar': emrat_kategoriye[parashikimet[i]],
                'pikët_besimi': pikët_besimi,
                'teksti_i_pastro': tekstet_e_pastra[i]
            })
        
        return jsonify({
            'rezultatet': rezultatet,
            'numri_i_teksteve': len(tekstet),
            'parashikime_me_sukses': len(rezultatet)
        })
        
    except Exception as e:
        return jsonify({'gabim': str(e)}), 500

@app.errorhandler(404)
def nuk_u_gjet(error):
    return jsonify({'gabim': 'Endpoint nuk u gjet'}), 404

@app.errorhandler(500)
def gabim_i_brendshem(error):
    return jsonify({'gabim': 'Gabim i brendshëm i serverit'}), 500

if __name__ == '__main__':
    print("Duke ngarkuar modelet e machine learning...")
    try:
        ngarko_modelet()
        print("Duke nisur serverin Flask...")
        print("API do të jetë i disponueshëm në: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Serveri nuk u nis: {e}")