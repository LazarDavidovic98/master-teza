import requests
import time

# Test popunjavanje ankete
test_data = {
    'godina_rodjenja': '1990',
    'drzava': 'Srbija',
    'strucna_sprema': 'visoka',
    'radni_odnos': 'da',
    'grana_oblast': 'Informacione tehnologije',
    'godine_staza': '5-10',
    'institucija': 'Test Company',
    'uza_oblast': 'Backend razvoj',
    'pisanje_softvera': 'cesto',
    'generativni_ai_poznavanje': '4',
    'programski_jezici': 'Python',
    'programska_okruzenja': 'VS Code',
    'poznati_ai_alati': ['chatgpt', 'github_copilot'],
    'ostalo_poznati_ai_alati': '',
    'svrhe_koriscenja': ['kod_pisanje', 'debugging'],
    'ostalo_svrhe': '',
    'poznavanje_principa': '3',
    'prompt_engineering': '3',
    'tehnicko_razumevanje': ['tokenizacija', 'context_window'],
    'problemi_ai': ['privacy', 'dependency'],
    'pravni_okviri': ['gdpr'],
    'provere_ai': ['manual_review'],
    'metode_evaluacije': ['functionality', 'code_quality'],
    'ogranicenja_ai': '3',
    'transformer_elementi': ['multi_head_attention'],
    'tehnike_treniranja': ['supervised_learning'],
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

# Pošalji POST zahtev
response = requests.post('http://127.0.0.1:5000/submit', data=test_data)
print(f"Status kod: {response.status_code}")
print(f"Response: {response.text}")
