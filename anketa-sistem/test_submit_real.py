import requests
import json

# URL aplikacije
url = "http://127.0.0.1:5000/submit"

# Test podaci za anketu
test_data = {
    'godina_rodjenja': '1995',
    'drzava': 'Srbija',
    'strucna_sprema': 'visoka',
    'radni_odnos': 'da',
    'grana_oblast': 'Informatika',
    'godine_staza': '5-10',
    'institucija': 'Microsoft',
    'uza_oblast': 'Data Science',
    'pisanje_softvera': 'da',
    'generativni_ai_poznavanje': 'da',
    'programski_jezici': 'da',
    'programska_okruzenja': 'da',
    'poznati_ai_alati': ['ChatGPT', 'GitHub Copilot'],
    'svrhe_koriscenja': ['Pisanje koda', 'Debugging', 'Učenje'],
    'poznavanje_principa': '4',
    'prompt_engineering': '3',
    'tehnicko_razumevanje': ['Tokenizacija', 'Context window'],
    'problemi_ai': ['Privatnost podataka', 'Bias u modelima'],
    'pravni_okviri': ['GDPR', 'AI Act'],
    'provere_ai': ['Validation', 'Peer review'],
    'metode_evaluacije': ['BLEU, ROUGE, METEOR', 'Human evaluation'],
    'ogranicenja_ai': 'tehnicke_limite',
    'transformer_elementi': ['Attention mechanism', 'Multi-head attention'],
    'tehnike_treniranja': ['Supervised learning', 'Transfer learning'],
    'koncepti_poznavanje': '3',
    # Kviz odgovori
    'chatgpt_omni': 'GPT-40',
    'copilot_task': 'Copilot Workspace', 
    'copilot_chat': 'Copilot X',
    'google_model': 'Gemini',
    'gpt_realtime': 'GPT-40',
    'codex_successor': 'GPT-3.5',
    'chatgpt_data_analysis': 'Advanced Data Analysis (Code Interpreter)',
    'copilot_workspace': 'Copilot Workspace',
    'anthropic_model': 'Claude',
    'creativity_parameter': 'Temperature',
    'transformer_basis': 'Transformeri',
    'university_guidelines': 'Stanford'
}

print("Šalje test anketu...")
try:
    response = requests.post(url, data=test_data)
    print(f"Status kod: {response.status_code}")
    print(f"Odgovor: {response.text}")
    if response.status_code == 200:
        print("✅ Anketa je uspešno poslata!")
    else:
        print("❌ Greška pri slanju ankete.")
except Exception as e:
    print(f"Greška: {e}")
