"""
Skripta za generisanje velikog broja test podataka u survey_responses.csv
Generiše 300 novih realističkih redova za testiranje sistema
"""

import csv
import random
from datetime import datetime, timedelta
import os

def generate_mass_data(num_records=300):
    """Generiši veliki broj test podataka za survey_responses.csv"""
    
    print(f"🚀 Generisanje {num_records} novih redova u survey_responses.csv...")
    
    # Realni podaci za generisanje
    countries = ['srbija', 'hrvatska', 'bosna_i_hercegovina', 'crna_gora', 'slovenija', 'severna_makedonija', 'bugarska', 'rumunija', 'nemacka', 'austrija']
    educations = ['srednja', 'visa', 'visoka', 'master', 'doktorat']
    employment = ['da', 'ne']
    experience_years = ['0', '1-2', '3-5', '6-10', '11-15', '16-20', '20+']
    institutions = ['Univerzitet u Beogradu', 'Univerzitet u Novom Sadu', 'Univerzitet u Nišu', 'Microsoft', 'Google', 'Amazon', 'Meta', 'Apple', 'IBM', 'Oracle', 'Comtrade', 'RT-RK', 'Nordeus']
    
    # Ciljane grupe
    target_groups = ['it_industrija', 'prosveta', 'medicina', 'kreativna_filmska', 'drustvene_nauke', 'ostalo']
    
    # IT Industrija podaci
    it_positions = ['Developer/Programer', 'Quality Assurance/Tester', 'DevOps Engineer', 'Project Manager', 'Business Analyst', 'UI/UX Designer', 'Data Scientist/Analyst']
    it_technologies = ['JavaScript/TypeScript', 'Python', 'Java', 'C#/.NET', 'React/Angular/Vue', 'Mobile (iOS/Android)', 'Cloud (AWS/Azure/GCP)']
    it_ai_usage = ['da_aktivno', 'povremeno', 'ne']
    
    # Prosveta podaci
    education_levels = ['Osnovno obrazovanje', 'Srednje obrazovanje', 'Više/Visoko obrazovanje', 'Predškolsko obrazovanje']
    subjects = ['Matematika', 'Informatika', 'Fizika', 'Hemija', 'Biologija', 'Srpski jezik', 'Engleski jezik', 'Istorija', 'Geografija', 'Likovna kultura']
    prosveta_ai_usage = ['da_redovno', 'povremeno', 'ne']
    
    # Medicina podaci
    medical_areas = ['opšta medicina', 'hirurgija', 'pedijatrija', 'kardiologija', 'neurologija', 'psihijatrija', 'onkologija', 'radiologija', 'farmakologija']
    medical_positions = ['Lekar/Doktor', 'Medicinska sestra/tehničar', 'Farmaceut', 'Administrator/Menadžer', 'Istraživač']
    medicina_ai_usage = ['da_u_praksi', 'da_u_istrazivanjima', 'da_u_administraciji', 'ne']
    
    # Kreativna industrija podaci
    creative_areas = ['Film i televizija', 'Grafički dizajn', 'Muzika i audio produkcija', 'Animacija i VFX', 'Gaming industrija', 'Fotografija', 'Pisanje/Kreativno pisanje']
    kreativna_ai_usage = ['da_aktivno', 'eksperimentisem', 'ne']
    
    # Društvene nauke podaci  
    social_sciences = ['sociologija', 'psihologija', 'antropologija', 'politikologija', 'filozofija', 'ekonomija', 'pedagogija', 'andragogija']
    social_activities = ['Naučno istraživanje', 'Nastava/Predavanje', 'Konsalting', 'Analiza podataka/Statistike']
    drustvene_ai_usage = ['da_u_istrazivanjima', 'da_u_analizi', 'ne']
    
    # Ostalo podaci
    other_areas = ['Poljoprivreda', 'Turizam', 'Saobraćaj', 'Trgovina', 'Ugostiteljstvo', 'Bankarstvo', 'Osiguranje', 'Konsalting', 'Državna uprava']
    ostalo_ai_usage = ['da', 'ne']
    ostalo_expertise = ['pocetnik', 'srednji', 'napredni']
    
    # AI alati
    ai_tools = ['ChatGPT', 'GitHub Copilot', 'Claude', 'Gemini', 'Bing Chat', 'Midjourney', 'DALL-E', 'Stable Diffusion', 'DeepL', 'Grammarly']
    
    # Svrhe korišćenja
    purposes = [
        'Pisanje teksta (eseja, mejlova, bloga)',
        'Pisanje koda',
        'Prevođenje jezika',
        'Kreiranje prezentacija',
        'Analiza podataka',
        'Učenje novih koncepata',
        'Brainstorming ideja',
        'Kreiranje slika'
    ]
    
    # Quiz odgovori - definišem tačne i netačne opcije
    quiz_options = {
        'chatgpt_omni': ['GPT-4', 'GPT-3.5', 'Claude', 'Gemini'],  # Tačan: GPT-4
        'copilot_task': ['Copilot Workspace', 'GitHub Copilot', 'Copilot X', 'Copilot Chat'],  # Tačan: Copilot Workspace  
        'copilot_chat': ['Copilot Chat', 'ChatGPT', 'Copilot X', 'GitHub Copilot'],
        'google_model': ['Gemini', 'PaLM', 'BERT', 'T5'],  # Tačan: Gemini
        'gpt_realtime': ['GPT-4', 'GPT-4.1', 'GPT-4 Turbo', 'GPT-5', 'GPT-4 Real-time'],
        'codex_successor': ['GPT-3.5', 'GitHub Copilot', 'Codex', 'GPT-4', 'Claude'],  # Tačan: GPT-3.5
        'chatgpt_data_analysis': ['Advanced Data Analysis (Code Interpreter)', 'ChatGPT Editor', 'Data Analyst', 'Code Interpreter'],
        'copilot_workspace': ['Copilot Workspace', 'Copilot X', 'GitHub Workspace', 'Copilot IDE'],  # Tačan: Copilot Workspace
        'anthropic_model': ['Claude', 'GPT-4', 'Gemini', 'PaLM'],  # Tačan: Claude
        'creativity_parameter': ['Temperature', 'Top-p', 'Attention heads', 'Learning rate'],  # Tačan: Temperature
        'transformer_basis': ['Transformeri', 'Recurrent Neural Networks (RNN)', 'Convolutional layers', 'LSTM'],  # Tačan: Transformeri
        'university_guidelines': ['Oxford', 'Harvard', 'MIT', 'Stanford']
    }
    
    # Učitaj postojeće podatke da vidim strukturu
    csv_file = 'survey_responses.csv'
    
    # Proverim da li fajl postoji
    if not os.path.exists(csv_file):
        print(f"❌ {csv_file} ne postoji!")
        return
    
    # Učitam postojeći header
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        existing_rows = list(reader)
    
    print(f"📄 Postojeći fajl ima {len(existing_rows)} redova")
    print(f"📋 Header: {len(header)} kolona")
    
    # Generiši nove redove
    new_rows = []
    
    for i in range(num_records):
        # Generiši timestamp u poslednjih 30 dana
        days_ago = random.randint(0, 30)
        timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Osnovni podaci
        birth_year = random.randint(1985, 2005)
        country = random.choice(countries)
        education = random.choice(educations)
        employment_status = random.choice(employment)
        target_group = random.choice(target_groups)
        experience = random.choice(experience_years)
        institution = random.choice(institutions) if employment_status == 'da' else ''
        
        # Generiši specifične podatke na osnovu ciljane grupe
        # Inicijalizuj sve sa praznim vrednostima
        it_pozicije = it_tehnologije = it_ai_koriscenje = ''
        prosveta_nivo = prosveta_predmeti = prosveta_ai_koriscenje = ''
        medicina_oblast = medicina_pozicije = medicina_ai_koriscenje = ''
        kreativna_oblasti = kreativna_ai_koriscenje = ''
        drustvene_oblast = drustvene_aktivnosti = drustvene_ai_koriscenje = ''
        ostalo_oblast = ostalo_ai_koriscenje = ostalo_ekspertiza = ''
        
        if target_group == 'it_industrija':
            # IT Industrija specifični podaci
            num_positions = random.randint(1, 3)
            selected_positions = random.sample(it_positions, min(num_positions, len(it_positions)))
            it_pozicije = ','.join(selected_positions)
            
            num_technologies = random.randint(1, 4)
            selected_technologies = random.sample(it_technologies, min(num_technologies, len(it_technologies)))
            it_tehnologije = ','.join(selected_technologies)
            
            it_ai_koriscenje = random.choice(it_ai_usage)
            
        elif target_group == 'prosveta':
            # Prosveta specifični podaci
            num_levels = random.randint(1, 2)
            selected_levels = random.sample(education_levels, min(num_levels, len(education_levels)))
            prosveta_nivo = ','.join(selected_levels)
            
            num_subjects = random.randint(1, 3)
            selected_subjects = random.sample(subjects, min(num_subjects, len(subjects)))
            prosveta_predmeti = ','.join(selected_subjects)
            
            prosveta_ai_koriscenje = random.choice(prosveta_ai_usage)
            
        elif target_group == 'medicina':
            # Medicina specifični podaci
            medicina_oblast = random.choice(medical_areas)
            
            num_positions = random.randint(1, 2)
            selected_positions = random.sample(medical_positions, min(num_positions, len(medical_positions)))
            medicina_pozicije = ','.join(selected_positions)
            
            medicina_ai_koriscenje = random.choice(medicina_ai_usage)
            
        elif target_group == 'kreativna_filmska':
            # Kreativna industrija specifični podaci
            num_areas = random.randint(1, 3)
            selected_areas = random.sample(creative_areas, min(num_areas, len(creative_areas)))
            kreativna_oblasti = ','.join(selected_areas)
            
            kreativna_ai_koriscenje = random.choice(kreativna_ai_usage)
            
        elif target_group == 'drustvene_nauke':
            # Društvene nauke specifični podaci
            drustvene_oblast = random.choice(social_sciences)
            
            num_activities = random.randint(1, 2)
            selected_activities = random.sample(social_activities, min(num_activities, len(social_activities)))
            drustvene_aktivnosti = ','.join(selected_activities)
            
            drustvene_ai_koriscenje = random.choice(drustvene_ai_usage)
            
        elif target_group == 'ostalo':
            # Ostalo specifični podaci
            ostalo_oblast = random.choice(other_areas)
            ostalo_ai_koriscenje = random.choice(ostalo_ai_usage)
            ostalo_ekspertiza = random.choice(ostalo_expertise)
        
        # AI alati - select 1-5 tools
        num_tools = random.randint(1, 5)
        selected_tools = random.sample(ai_tools, min(num_tools, len(ai_tools)))
        tools_str = ','.join(selected_tools)
        
        # Ostalo AI tools
        other_tools = random.choice(['', 'Perplexity', 'Notion AI', 'Jasper', 'Copy.ai'])
        
        # Purposes - select 1-4 purposes
        num_purposes = random.randint(1, 4)
        selected_purposes = random.sample(purposes, min(num_purposes, len(purposes)))
        purposes_str = ','.join(selected_purposes)
        
        # Ostale svrhe
        other_purposes = random.choice(['', 'Kreiranje muzike', 'Video editing', 'Game development'])
        
        # Knowledge levels (scale answers)
        principle_knowledge = random.choice(['nimalo', 'malo', 'umerno', 'dosta', 'veoma_dosta'])
        prompt_engineering = random.choice(['da_praktikujem', 'da_cuo', 'ne'])
        
        technical_understanding = random.choice([
            'Arhitektura Transformer modela',
            'Large Language Models (LLM)',
            'Fine-tuning modela na sopstvenim podacima',
            'Rad sa embedding vektorima',
            'Ne razumem nijedan od navedenih'
        ])
        
        ai_problems = random.choice([
            'Privatnost podataka',
            'Autorska prava i plagijati',
            'Generisanje lažnih informacija',
            'Pristrasnost u odgovorima',
            'Zavisnost od AI alata'
        ])
        
        legal_framework = random.choice(['AI Act (EU)', 'GDPR', 'Ne znam za specifične zakone', 'Ostalo'])
        
        ai_verification = random.choice([
            'Može da napravi SQL upit optimizovan za performanse',
            'Može da generiše unit testove za kod',
            'Može da objasni složen algoritam',
            'Može da generiše podatke za analizu u CSV/JSON formatu',
            'Može da se poveže sa API-jem (npr. Flask, Django, FastAPI)'
        ])
        
        eval_methods = random.choice(['BLEU, ROUGE, METEOR', 'Human evaluation', 'Perplexity', 'A/B testiranje', 'Ne znam'])
        
        ai_limitations = random.choice([
            'ne_razlikujem',
            'osnovna_ogranicenja',
            'tehnicke_limite'
        ])
        
        transformer_elements = random.choice(['Encoder', 'Decoder', 'Self-attention', 'Multi-head attention', 'Ne znam'])
        
        training_techniques = random.choice([
            'Supervised learning na velikim korpusima teksta',
            'Unsupervised learning (autoregresivno modelovanje)',
            'Reinforcement Learning from Human Feedback (RLHF)',
            'Fine-tuning na domen-specifičnim podacima',
            'Ne znam'
        ])
        
        concepts_knowledge = random.choice(['agent', 'task', 'edit', 'plan_execute', 'ne_poznajem'])
        
        # Quiz odgovori - generiši sa realnošću (70% tačnih, 30% netačnih)
        quiz_answers = {}
        for question, options in quiz_options.items():
            if random.random() < 0.7:  # 70% verovatnoća za tačan odgovor
                # Za tačne odgovore koristim eksplicitno definisane tačne vrednosti
                if question == 'chatgpt_omni':
                    quiz_answers[question] = 'GPT-4'
                elif question == 'copilot_task':
                    quiz_answers[question] = 'Copilot Workspace'
                elif question == 'copilot_chat':
                    quiz_answers[question] = 'Copilot X'
                elif question == 'google_model':
                    quiz_answers[question] = 'Gemini'
                elif question == 'gpt_realtime':
                    quiz_answers[question] = 'GPT-4'
                elif question == 'gpt_realtime':
                    quiz_answers[question] = 'GPT-4'
                elif question == 'codex_successor':
                    quiz_answers[question] = 'GPT-3.5'
                elif question == 'chatgpt_data_analysis':
                    quiz_answers[question] = 'Advanced Data Analysis (Code Interpreter)'
                elif question == 'copilot_workspace':
                    quiz_answers[question] = 'Copilot Workspace'
                elif question == 'anthropic_model':
                    quiz_answers[question] = 'Claude'
                elif question == 'creativity_parameter':
                    quiz_answers[question] = 'Temperature'
                elif question == 'transformer_basis':
                    quiz_answers[question] = 'Transformeri'
                elif question == 'university_guidelines':
                    quiz_answers[question] = 'Stanford'
                else:
                    quiz_answers[question] = options[0]  # Prvi je uvek tačan
            else:
                # 30% netačnih odgovora - bira nasumičnu opciju različitu od tačne
                correct_answers = {
                    'chatgpt_omni': 'GPT-4',
                    'copilot_task': 'Copilot Workspace', 
                    'copilot_chat': 'Copilot X',
                    'google_model': 'Gemini',
                    'gpt_realtime': 'GPT-4',
                    'codex_successor': 'GPT-3.5',
                    'chatgpt_data_analysis': 'Advanced Data Analysis (Code Interpreter)',
                    'copilot_workspace': 'Copilot Workspace',
                    'anthropic_model': 'Claude',
                    'creativity_parameter': 'Temperature',
                    'transformer_basis': 'Transformeri',
                    'university_guidelines': 'Stanford'
                }
                wrong_options = [opt for opt in options if opt != correct_answers.get(question)]
                quiz_answers[question] = random.choice(wrong_options) if wrong_options else random.choice(options)
        
        # Kreiraj red prema novoj strukturi header-a
        row = [
            timestamp,                                    # timestamp
            str(birth_year),                             # godina_rodjenja  
            country,                                     # drzava
            education,                                   # strucna_sprema
            employment_status,                           # radni_odnos
            target_group,                                # ciljana_grupa
            experience,                                  # godine_staza
            institution,                                 # institucija
            # IT Industrija fields
            it_pozicije,                                 # it_pozicije
            it_tehnologije,                              # it_tehnologije
            it_ai_koriscenje,                            # it_ai_koriscenje
            # Prosveta fields
            prosveta_nivo,                               # prosveta_nivo
            prosveta_predmeti,                           # prosveta_predmeti
            prosveta_ai_koriscenje,                      # prosveta_ai_koriscenje
            # Medicina fields
            medicina_oblast,                             # medicina_oblast
            medicina_pozicije,                           # medicina_pozicije
            medicina_ai_koriscenje,                      # medicina_ai_koriscenje
            # Kreativna industrija fields
            kreativna_oblasti,                           # kreativna_oblasti
            kreativna_ai_koriscenje,                     # kreativna_ai_koriscenje
            # Društvene nauke fields
            drustvene_oblast,                            # drustvene_oblast
            drustvene_aktivnosti,                        # drustvene_aktivnosti
            drustvene_ai_koriscenje,                     # drustvene_ai_koriscenje
            # Ostalo fields
            ostalo_oblast,                               # ostalo_oblast
            ostalo_ai_koriscenje,                        # ostalo_ai_koriscenje
            ostalo_ekspertiza,                           # ostalo_ekspertiza
            # Rest of existing fields
            tools_str,                                   # poznati_ai_alati
            other_tools,                                 # ostalo_poznati_ai_alati
            purposes_str,                                # svrhe_koriscenja
            other_purposes,                              # ostalo_svrhe
            principle_knowledge,                         # poznavanje_principa
            prompt_engineering,                          # prompt_engineering
            technical_understanding,                     # tehnicko_razumevanje
            ai_problems,                                 # problemi_ai
            legal_framework,                             # pravni_okviri
            ai_verification,                             # provere_ai
            eval_methods,                                # metode_evaluacije
            ai_limitations,                              # ogranicenja_ai
            transformer_elements,                        # transformer_elementi
            training_techniques,                         # tehnike_treniranja
            concepts_knowledge,                          # koncepti_poznavanje
            quiz_answers.get('chatgpt_omni', ''),        # chatgpt_omni
            quiz_answers.get('copilot_task', ''),        # copilot_task
            quiz_answers.get('copilot_chat', ''),        # copilot_chat
            quiz_answers.get('google_model', ''),        # google_model
            quiz_answers.get('gpt_realtime', ''),        # gpt_realtime
            quiz_answers.get('codex_successor', ''),     # codex_successor
            quiz_answers.get('chatgpt_data_analysis', ''), # chatgpt_data_analysis
            quiz_answers.get('copilot_workspace', ''),   # copilot_workspace
            quiz_answers.get('anthropic_model', ''),     # anthropic_model
            quiz_answers.get('creativity_parameter', ''), # creativity_parameter
            quiz_answers.get('transformer_basis', ''),   # transformer_basis
            quiz_answers.get('university_guidelines', ''), # university_guidelines
        ]
        
        new_rows.append(row)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  ✅ Generirano {i + 1}/{num_records} redova...")
    
    # Sačuvaj sve u fajl
    print(f"💾 Upisivanje {len(new_rows)} novih redova u {csv_file}...")
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Zadrži postojeći header
        writer.writerows(existing_rows)  # Zadrži postojeće redove
        writer.writerows(new_rows)  # Dodaj nove redove
    
    total_rows = len(existing_rows) + len(new_rows)
    print(f"🎯 Uspešno! survey_responses.csv sada ima {total_rows} redova")
    print(f"   - Postojeći redovi: {len(existing_rows)}")
    print(f"   - Novi redovi: {len(new_rows)}")
    
    # Proveri da li je file watcher aktivan
    print(f"\n⚡ File watcher će automatski detektovati promene i regenerisati sve analize!")
    print(f"🔍 Proverite Flask aplikaciju na http://127.0.0.1:5000/data_science")

def main():
    """Main funkcija"""
    print("🔬 MASS DATA GENERATOR za Survey Responses")
    print("="*50)
    
    # Pitaj korisnika koliko redova želi
    try:
        num_records = input("\n📊 Koliko novih redova da generiram (preporučeno 50-500)? [300]: ").strip()
        num_records = int(num_records) if num_records else 300
        
        if num_records <= 0:
            print("❌ Broj redova mora biti pozitivan!")
            return
            
        if num_records > 1000:
            confirm = input(f"⚠️  {num_records} je puno redova. Sigurni ste? (da/ne): ").lower()
            if confirm != 'da':
                print("❌ Otkazano.")
                return
        
        # Generiši podatke
        generate_mass_data(num_records)
        
        print(f"\n✨ Sledeći koraci:")
        print(f"  1. Pokrenite Flask aplikaciju: python app.py")
        print(f"  2. Idite na: http://127.0.0.1:5000/data_science")
        print(f"  3. File watcher će automatski regenerisati sve fajlove!")
        
    except ValueError:
        print("❌ Molim unesite valjan broj!")
    except KeyboardInterrupt:
        print("\n❌ Prekinuto od strane korisnika.")
    except Exception as e:
        print(f"❌ Greška: {e}")

if __name__ == "__main__":
    main()
