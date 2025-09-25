#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import random
from datetime import datetime, timedelta

# Definišem sve opcije za simulaciju
GODINE_RODJENJA = ['1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005']
DRZAVE = ['Srbija', 'Hrvatska', 'Bosna i Hercegovina', 'Crna Gora', 'Slovenija', 'Makedonija', 'Nemačka', 'Austrija']
STRUCNA_SPREMA = ['srednja', 'visa', 'visoka', 'master', 'doktorat']
RADNI_ODNOS = ['da', 'ne']
GRANE_OBLASTI = ['Informacione tehnologije', 'Ekonomija', 'Mašinstvo', 'Elektrotehnika', 'Matematika', 'Fizika', 'Drugo']
GODINE_STAZA = ['0-1', '1-3', '3-5', '5-10', '10+']
UZA_OBLASTI = ['Frontend razvoj', 'Backend razvoj', 'Data Science', 'AI/ML', 'DevOps', 'Mobile razvoj', 'Drugo']
PISANJE_SOFTVERA = ['nikad', 'retko', 'ponekad', 'cesto', 'uvek']
GENERATIVNI_AI = ['1', '2', '3', '4', '5']
PROGRAMSKI_JEZICI = ['Python', 'JavaScript', 'Java', 'C#', 'C++', 'PHP', 'Go', 'Ruby']
PROGRAMSKA_OKRUZENJA = ['VS Code', 'IntelliJ IDEA', 'Visual Studio', 'PyCharm', 'Eclipse', 'Vim/Neovim']

# AI alati opcije
AI_ALATI_OPTIONS = ['chatgpt', 'github_copilot', 'claude', 'gemini', 'copilot_chat', 'tabnine', 'poznajem_ostalo']
SVRHE_OPTIONS = ['kod_pisanje', 'kod_objasnjavanje', 'debugging', 'dokumentacija', 'ucenje', 'testiranje', 'svrha_ostalo']
POZNAVANJE_PRINCIPA = ['1', '2', '3', '4', '5']
PROMPT_ENGINEERING = ['1', '2', '3', '4', '5']

# Tehničko razumevanje opcije
TEHNICKO_RAZUMEVANJE_OPTIONS = ['tokenizacija', 'attention_mechanism', 'fine_tuning', 'few_shot_learning', 'context_window', 'embeddings']

# Problemi AI opcije  
PROBLEMI_AI_OPTIONS = ['hallucinations', 'bias', 'privacy', 'job_displacement', 'misinformation', 'dependency']

# Pravni okviri opcije
PRAVNI_OKVIRI_OPTIONS = ['gdpr', 'ai_act', 'copyright', 'liability', 'transparency', 'accountability']

# Provere AI opcije
PROVERE_AI_OPTIONS = ['manual_review', 'automated_testing', 'peer_review', 'version_control', 'documentation', 'validation']

# Metode evaluacije opcije
METODE_EVALUACIJE_OPTIONS = ['code_quality', 'performance', 'security', 'functionality', 'maintainability', 'user_feedback']

OGRANICENJA_AI = ['1', '2', '3', '4', '5']

# Transformer elementi opcije
TRANSFORMER_ELEMENTI_OPTIONS = ['attention_heads', 'positional_encoding', 'layer_normalization', 'feed_forward', 'residual_connections', 'multi_head_attention']

# Tehnike treniranja opcije
TEHNIKE_TRENIRANJA_OPTIONS = ['supervised_learning', 'unsupervised_learning', 'reinforcement_learning', 'transfer_learning', 'few_shot', 'zero_shot']

KONCEPTI_POZNAVANJE = ['1', '2', '3', '4', '5']

# Kviz odgovori (neki tačni, neki netačni za realizam)
QUIZ_ANSWERS = {
    'chatgpt_omni': ['GPT-3.5', 'GPT-4', 'GPT-4.1', 'GPT-40', 'GPT-5', 'ne_znam'],
    'copilot_task': ['Copilot 2021', 'Copilot X', 'Copilot Workspace', 'Copilot Agents'],
    'copilot_chat': ['Copilot 2021', 'Copilot X', 'Copilot Workspace', 'Copilot Agents'],
    'google_model': ['Claude 3 Opus', 'Claude 3 Sonnet', 'Claude 3.5 Sonnet', 'Gemini', 'LLaMA'],
    'gpt_realtime': ['GPT-3.5', 'GPT-4', 'GPT-4.1', 'GPT-40', 'GPT-5', 'ne_znam'],
    'codex_successor': ['GPT-2', 'GPT-3', 'GPT-3.5', 'Codex', 'GPT-4', 'ne_znam'],
    'chatgpt_data_analysis': ['ChatGPT Editor', 'ChatGPT Agent', 'Advanced Data Analysis (Code Interpreter)', 'ChatGPT Task Runner', 'Copilot Workspace', 'ne_znam'],
    'copilot_workspace': ['Copilot 2021', 'Copilot X', 'Copilot Workspace', 'Copilot Agents', 'ne_znam'],
    'anthropic_model': ['Claude', 'Gemini', 'Codex', 'LLaMA', 'ne_znam'],
    'creativity_parameter': ['Learning rate', 'Attention heads', 'Batch size', 'Temperature', 'Dropout', 'ne_znam'],
    'transformer_basis': ['Recurrent Neural Networks (RNN)', 'Support Vector Machines (SVM)', 'Transformeri', 'Decision Trees', 'K-means clustering', 'ne_znam'],
    'university_guidelines': ['MIT', 'Oxford', 'Sorbonne', 'Stanford', 'Univerzitet u Beogradu', 'ne_znam']
}

def generate_realistic_response():
    """Generiše realistične odgovore za anketu"""
    
    # Osnovne informacije
    godina_rodjenja = random.choice(GODINE_RODJENJA)
    drzava = random.choice(DRZAVE)
    strucna_sprema = random.choice(STRUCNA_SPREMA)
    radni_odnos = random.choice(RADNI_ODNOS)
    grana_oblast = random.choice(GRANE_OBLASTI)
    godine_staza = random.choice(GODINE_STAZA)
    
    # Institucija samo ako radi
    institucija = ''
    if radni_odnos == 'da':
        institucije = ['Univerzitet u Beogradu', 'Novi Sad Tech', 'Microsoft Srbija', 'Google', 'Facebook', 'Amazon', 'Tesla', 'Startap XYZ']
        institucija = random.choice(institucije)
    
    uza_oblast = random.choice(UZA_OBLASTI)
    pisanje_softvera = random.choice(PISANJE_SOFTVERA)
    generativni_ai_poznavanje = random.choice(GENERATIVNI_AI)
    programski_jezici = random.choice(PROGRAMSKI_JEZICI)
    programska_okruzenja = random.choice(PROGRAMSKA_OKRUZENJA)
    
    # AI alati - biraj 2-4 opcije
    poznati_ai_alati = random.sample(AI_ALATI_OPTIONS[:-1], random.randint(2, 4))
    ostalo_poznati_ai_alati = ''
    if 'poznajem_ostalo' in poznati_ai_alati:
        ostalo_poznati_ai_alati = random.choice(['Cursor', 'Replit AI', 'Codeium', 'Amazon CodeWhisperer'])
    
    # Svrhe - biraj 2-5 opcija
    svrhe_koriscenja = random.sample(SVRHE_OPTIONS[:-1], random.randint(2, 5))
    ostalo_svrhe = ''
    if 'svrha_ostalo' in svrhe_koriscenja:
        ostalo_svrhe = random.choice(['Code review', 'Refactoring', 'Architecture planning'])
    
    poznavanje_principa = random.choice(POZNAVANJE_PRINCIPA)
    prompt_engineering = random.choice(PROMPT_ENGINEERING)
    
    # Tehničko razumevanje - biraj 2-4 opcije
    tehnicko_razumevanje = random.sample(TEHNICKO_RAZUMEVANJE_OPTIONS, random.randint(2, 4))
    
    # Problemi AI - biraj 2-4 opcije
    problemi_ai = random.sample(PROBLEMI_AI_OPTIONS, random.randint(2, 4))
    
    # Pravni okviri - biraj 1-3 opcije
    pravni_okviri = random.sample(PRAVNI_OKVIRI_OPTIONS, random.randint(1, 3))
    
    # Provere AI - biraj 2-4 opcije
    provere_ai = random.sample(PROVERE_AI_OPTIONS, random.randint(2, 4))
    
    # Metode evaluacije - biraj 2-4 opcije
    metode_evaluacije = random.sample(METODE_EVALUACIJE_OPTIONS, random.randint(2, 4))
    
    ogranicenja_ai = random.choice(OGRANICENJA_AI)
    
    # Transformer elementi - biraj 2-4 opcije
    transformer_elementi = random.sample(TRANSFORMER_ELEMENTI_OPTIONS, random.randint(2, 4))
    
    # Tehnike treniranja - biraj 2-3 opcije
    tehnike_treniranja = random.sample(TEHNIKE_TRENIRANJA_OPTIONS, random.randint(2, 3))
    
    koncepti_poznavanje = random.choice(KONCEPTI_POZNAVANJE)
    
    # Kviz odgovori - mešavina tačnih i netačnih odgovora
    quiz_responses = {}
    for question, options in QUIZ_ANSWERS.items():
        # 60% šanse za tačan odgovor, 25% za netačan, 15% za "ne znam"
        rand = random.random()
        if rand < 0.15:  # Ne znam
            quiz_responses[question] = 'ne_znam'
        elif rand < 0.6:  # Pokušaj tačnog odgovora
            # Tačni odgovori prema implementaciji
            correct_answers = {
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
            quiz_responses[question] = correct_answers[question]
        else:  # Netačan odgovor
            wrong_options = [opt for opt in options if opt != 'ne_znam']
            correct_answers = {
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
            wrong_options = [opt for opt in wrong_options if opt != correct_answers[question]]
            if wrong_options:
                quiz_responses[question] = random.choice(wrong_options)
            else:
                quiz_responses[question] = correct_answers[question]
    
    return {
        'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30), 
                                                hours=random.randint(0, 23), 
                                                minutes=random.randint(0, 59))).strftime('%Y-%m-%d %H:%M:%S'),
        'godina_rodjenja': godina_rodjenja,
        'drzava': drzava,
        'strucna_sprema': strucna_sprema,
        'radni_odnos': radni_odnos,
        'grana_oblast': grana_oblast,
        'godine_staza': godine_staza,
        'institucija': institucija,
        'uza_oblast': uza_oblast,
        'pisanje_softvera': pisanje_softvera,
        'generativni_ai_poznavanje': generativni_ai_poznavanje,
        'programski_jezici': programski_jezici,
        'programska_okruzenja': programska_okruzenja,
        'poznati_ai_alati': ','.join(poznati_ai_alati),
        'ostalo_poznati_ai_alati': ostalo_poznati_ai_alati,
        'svrhe_koriscenja': ','.join(svrhe_koriscenja),
        'ostalo_svrhe': ostalo_svrhe,
        'poznavanje_principa': poznavanje_principa,
        'prompt_engineering': prompt_engineering,
        'tehnicko_razumevanje': ','.join(tehnicko_razumevanje),
        'problemi_ai': ','.join(problemi_ai),
        'pravni_okviri': ','.join(pravni_okviri),
        'provere_ai': ','.join(provere_ai),
        'metode_evaluacije': ','.join(metode_evaluacije),
        'ogranicenja_ai': ogranicenja_ai,
        'transformer_elementi': ','.join(transformer_elementi),
        'tehnike_treniranja': ','.join(tehnike_treniranja),
        'koncepti_poznavanje': koncepti_poznavanje,
        'chatgpt_omni': quiz_responses['chatgpt_omni'],
        'copilot_task': quiz_responses['copilot_task'],
        'copilot_chat': quiz_responses['copilot_chat'],
        'google_model': quiz_responses['google_model'],
        'gpt_realtime': quiz_responses['gpt_realtime'],
        'codex_successor': quiz_responses['codex_successor'],
        'chatgpt_data_analysis': quiz_responses['chatgpt_data_analysis'],
        'copilot_workspace': quiz_responses['copilot_workspace'],
        'anthropic_model': quiz_responses['anthropic_model'],
        'creativity_parameter': quiz_responses['creativity_parameter'],
        'transformer_basis': quiz_responses['transformer_basis'],
        'university_guidelines': quiz_responses['university_guidelines']
    }

def main():
    """Glavna funkcija za kreiranje CSV fajla sa simuliranim odgovorima"""
    
    # Kreiraj CSV fajl sa header-ima
    headers = [
        'timestamp',
        'godina_rodjenja', 
        'drzava',
        'strucna_sprema',
        'radni_odnos',
        'grana_oblast',
        'godine_staza',
        'institucija',
        'uza_oblast',
        'pisanje_softvera',
        'generativni_ai_poznavanje',
        'programski_jezici',
        'programska_okruzenja',
        'poznati_ai_alati',
        'ostalo_poznati_ai_alati',
        'svrhe_koriscenja',
        'ostalo_svrhe',
        'poznavanje_principa',
        'prompt_engineering',
        'tehnicko_razumevanje',
        'problemi_ai',
        'pravni_okviri',
        'provere_ai',
        'metode_evaluacije',
        'ogranicenja_ai',
        'transformer_elementi',
        'tehnike_treniranja',
        'koncepti_poznavanje',
        'chatgpt_omni',
        'copilot_task',
        'copilot_chat',
        'google_model',
        'gpt_realtime',
        'codex_successor',
        'chatgpt_data_analysis',
        'copilot_workspace',
        'anthropic_model',
        'creativity_parameter',
        'transformer_basis',
        'university_guidelines'
    ]
    
    with open('survey_responses.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Zapiši header
        writer.writerow(headers)
        
        # Generiši 20 simuliranih odgovora
        for i in range(20):
            response = generate_realistic_response()
            row = [response[header] for header in headers]
            writer.writerow(row)
            print(f"Generisan odgovor {i+1}/20")
    
    print(f"\n✅ Uspešno generisano 20 simuliranih odgovora u survey_responses.csv!")
    print(f"📊 Fajl sadrži {len(headers)} kolona uključujući sve kviz odgovore")

if __name__ == "__main__":
    main()
