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
    countries = ['srbija', 'hrvatska', 'bosna_hercegovina', 'montenegro', 'slovenija', 'makedonija', 'bugarska', 'rumunija']
    educations = ['srednja', 'bachelor', 'master', 'phd', 'vss']
    employment = ['da', 'ne', 'student', 'penzioner']
    fields = ['Elektrotehnika', 'Računarstvo', 'Matematika', 'Fizika', 'Mašinstvo', 'Građevinarstvo', 'Ekonomija', 'Medicina']
    experience_years = ['0-1', '1-3', '3-5', '5-10', '10-15', '15+']
    institutions = ['Univerzitet u Beogradu', 'Univerzitet u Novom Sadu', 'Univerzitet u Nišu', 'FON', 'ETF', 'PMF', 'FTN', 'Singidunum', 'Megatrend']
    specializations = ['Softversko inžinjerstvo', 'Veštačka inteligencija', 'Računarske mreže', 'Baze podataka', 'Web development', 'Mobilno programiranje', 'Finansije', 'Marketing']
    
    # Programski jezici i okruženja
    prog_languages = ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'PHP', 'Ruby', 'Go', 'Rust', 'TypeScript']
    environments = ['Visual Studio Code', 'IntelliJ IDEA', 'Eclipse', 'PyCharm', 'Visual Studio', 'Atom', 'Sublime Text', 'Vim', 'Emacs']
    
    # AI alati
    ai_tools = ['ChatGPT', 'GitHub Copilot', 'Claude', 'Gemini', 'Bing Chat', 'Midjourney', 'DALL-E', 'Stable Diffusion', 'DeepL', 'Grammarly']
    
    # Svrhe korišćenja
    purposes = [
        'Pisanje koda',
        'Debugging',
        'Pisanje dokumentacije', 
        'Pisanje teksta (eseja, mejlova, bloga)',
        'Kreiranje prezentacija',
        'Prevod teksta',
        'Analiza podataka',
        'Kreiranje slika',
        'Učenje novih koncepata',
        'Brainstorming ideja'
    ]
    
    # Quiz odgovori - definišem tačne i netačne opcije
    quiz_options = {
        'chatgpt_omni': ['GPT-4', 'GPT-3.5', 'Claude', 'Gemini'],  # Tačan: GPT-4
        'copilot_task': ['Copilot Workspace', 'GitHub Copilot', 'Copilot X', 'Copilot Chat'],  # Tačan: Copilot Workspace  
        'copilot_chat': ['Copilot Chat', 'ChatGPT', 'Copilot X', 'GitHub Copilot'],
        'google_model': ['Gemini', 'PaLM', 'BERT', 'T5'],  # Tačan: Gemini
        'gpt_realtime': ['GPT-4.1', 'GPT-4 Turbo', 'GPT-5', 'GPT-4 Real-time'],
        'codex_successor': ['GitHub Copilot', 'Codex', 'GPT-4', 'Claude'],  # Tačan: GitHub Copilot
        'chatgpt_data_analysis': ['ChatGPT Editor', 'Advanced Data Analysis', 'Code Interpreter', 'Data Analyst'],
        'copilot_workspace': ['Copilot Workspace', 'Copilot X', 'GitHub Workspace', 'Copilot IDE'],  # Tačan: Copilot Workspace
        'anthropic_model': ['Claude', 'GPT-4', 'Gemini', 'PaLM'],  # Tačan: Claude
        'creativity_parameter': ['Temperature', 'Top-p', 'Attention heads', 'Learning rate'],  # Tačan: Temperature
        'transformer_basis': ['Attention heads', 'Recurrent Neural Networks (RNN)', 'Convolutional layers', 'LSTM'],  # Tačan: Attention heads
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
        
        # Generiši realne podatke
        birth_year = random.randint(1985, 2005)
        country = random.choice(countries)
        education = random.choice(educations)
        employment_status = random.choice(employment)
        field = random.choice(fields)
        experience = random.choice(experience_years)
        institution = random.choice(institutions)
        specialization = random.choice(specializations)
        
        # Software writing experience (1-10)
        software_exp = random.choice(['da', 'ne'])
        ai_knowledge = random.choice(['da', 'ne'])
        
        # Programming languages - random selection
        selected_languages = random.sample(prog_languages, random.randint(1, 4))
        languages_str = random.choice(['da', 'ne']) if random.random() > 0.3 else 'da'
        
        # Programming environments
        environments_str = random.choice(['da', 'ne']) if random.random() > 0.3 else 'da'
        
        # AI tools - select 1-5 tools
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
        prompt_engineering = random.choice(['da', 'ne'])
        technical_understanding = random.choice([
            'Arhitektura Transformer modela',
            'Large Language Models (LLM)',
            'Fine-tuning modela',
            'Ne razumem nijedan od navedenih'
        ])
        
        ai_problems = random.choice([
            'Privatnost podataka',
            'Pristrasnost AI',
            'Halucinations (netačni odgovori)',
            'Intelectualna svojina'
        ])
        
        legal_framework = random.choice(['AI Act (EU)', 'GDPR', 'Ne znam', 'Ostalo'])
        
        ai_verification = random.choice([
            'Može da napravi SQL upit optimizovan za performanse',
            'Može da generiše unit testove za kod',
            'Može da objasni algoritam',
            'Nijedan od navedenih'
        ])
        
        eval_methods = random.choice(['BLEU, ROUGE, METEOR', 'Human evaluation', 'Perplexity', 'Ne znam'])
        ai_limitations = random.choice(['da', 'ne', 'ne_razlikujem'])
        
        transformer_elements = random.choice(['Encoder', 'Decoder', 'Attention mechanism', 'Ne znam'])
        training_techniques = random.choice([
            'Supervised learning',
            'Unsupervised learning', 
            'Reinforcement learning',
            'Masked language modeling'
        ])
        
        concepts_knowledge = random.choice(['agent', 'prompt', 'token', 'embedding'])
        
        # Quiz odgovori - generiši sa realnošću (70% tačnih, 30% netačnih)
        quiz_answers = {}
        for question, options in quiz_options.items():
            if random.random() < 0.7:  # 70% verovatnoća za tačan odgovor
                # Za tačne odgovore koristim prvi u listi (koji su označeni kao tačni)
                if question == 'chatgpt_omni':
                    quiz_answers[question] = 'GPT-4'
                elif question == 'copilot_task':
                    quiz_answers[question] = 'Copilot Workspace'
                elif question == 'google_model':
                    quiz_answers[question] = 'Gemini'
                elif question == 'codex_successor':
                    quiz_answers[question] = 'GitHub Copilot'
                elif question == 'anthropic_model':
                    quiz_answers[question] = 'Claude'
                elif question == 'creativity_parameter':
                    quiz_answers[question] = 'Temperature'
                elif question == 'transformer_basis':
                    quiz_answers[question] = 'Attention heads'
                else:
                    quiz_answers[question] = random.choice(options)
            else:
                # 30% netačnih odgovora
                quiz_answers[question] = random.choice(options)
        
        # Kreiraj red prema strukturi header-a
        row = [
            timestamp,                                    # timestamp
            str(birth_year),                             # godina_rodjenja  
            country,                                     # drzava
            education,                                   # strucna_sprema
            employment_status,                           # radni_odnos
            field,                                       # grana_oblast
            experience,                                  # godine_staza
            institution,                                 # institucija
            specialization,                              # uza_oblast
            software_exp,                               # pisanje_softvera
            ai_knowledge,                               # generativni_ai_poznavanje
            languages_str,                              # programski_jezici
            environments_str,                           # programska_okruzenja
            tools_str,                                  # poznati_ai_alati
            other_tools,                                # ostalo_poznati_ai_alati
            purposes_str,                               # svrhe_koriscenja
            other_purposes,                             # ostalo_svrhe
            principle_knowledge,                        # poznavanje_principa
            prompt_engineering,                         # prompt_engineering
            technical_understanding,                    # tehnicko_razumevanje
            ai_problems,                                # problemi_ai
            legal_framework,                            # pravni_okviri
            ai_verification,                            # provere_ai
            eval_methods,                               # metode_evaluacije
            ai_limitations,                             # ogranicenja_ai
            transformer_elements,                       # transformer_elementi
            training_techniques,                        # tehnike_treniranja
            concepts_knowledge,                         # koncepti_poznavanje
            quiz_answers.get('chatgpt_omni', ''),       # chatgpt_omni
            quiz_answers.get('copilot_task', ''),       # copilot_task
            quiz_answers.get('copilot_chat', ''),       # copilot_chat
            quiz_answers.get('google_model', ''),       # google_model
            quiz_answers.get('gpt_realtime', ''),       # gpt_realtime
            quiz_answers.get('codex_successor', ''),    # codex_successor
            quiz_answers.get('chatgpt_data_analysis', ''), # chatgpt_data_analysis
            quiz_answers.get('copilot_workspace', ''),  # copilot_workspace
            quiz_answers.get('anthropic_model', ''),    # anthropic_model
            quiz_answers.get('creativity_parameter', ''), # creativity_parameter
            quiz_answers.get('transformer_basis', ''),  # transformer_basis
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
