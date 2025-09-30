from flask import Flask, render_template, request, redirect, url_for, flash, session
import csv
import os
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import base64
from io import BytesIO, StringIO

# Import our data science extension
try:
    from data_science_extension import DataScienceManager
    from advanced_analytics import AdvancedAnalytics
    from file_watcher import FileWatcher
    import networkx as nx
    DATA_SCIENCE_AVAILABLE = True
except ImportError:
    DATA_SCIENCE_AVAILABLE = False
    print("Data Science extension not available - running in basic mode")

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Initialize file watcher for automatic updates
file_watcher = None
if DATA_SCIENCE_AVAILABLE:
    try:
        data_manager = DataScienceManager()
        file_watcher = FileWatcher(data_manager)
        file_watcher.start()
    except Exception as e:
        print(f"File Watcher initialization failed: {e}")
        file_watcher = None

# Cleanup function for graceful shutdown
import atexit

def cleanup():
    """Cleanup function called on app shutdown"""
    if file_watcher and file_watcher.is_running():
        print("🛑 Zaustavljam File Watcher...")
        file_watcher.stop()

atexit.register(cleanup)

# File path for storing survey responses
RESPONSES_FILE = 'survey_responses.csv'
def load_survey_data():
    """Učitaj podatke iz ankete kao pandas DataFrame"""
    if not os.path.exists(RESPONSES_FILE):
        return pd.DataFrame()
    
    try:
        # Pokušaj prvo sa normalnim čitanjem
        df = pd.read_csv(RESPONSES_FILE, encoding='utf-8')
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except pd.errors.ParserError as e:
        print(f"Greška pri parsiranju CSV-a: {e}")
        print("Pokušavam sa naprednim čitanjem...")
        
        try:
            # Pokušaj sa error_bad_lines=False da preskočiš problematične linije
            df = pd.read_csv(RESPONSES_FILE, encoding='utf-8', on_bad_lines='skip')
            print(f"Uspešno učitano {len(df)} redova sa preskakanjem loših linija.")
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e2:
            print(f"I dalje greška: {e2}")
            
            # Poslednji pokušaj - ručno čitanje i čišćenje
            try:
                with open(RESPONSES_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Uzmi header
                header = lines[0].strip()
                expected_cols = len(header.split(','))
                print(f"Header ima {expected_cols} kolona")
                
                # Filtrriraj samo validne linije
                clean_lines = [header]
                for i, line in enumerate(lines[1:], 1):
                    cols = line.count(',') + 1
                    if cols == expected_cols:
                        clean_lines.append(line.strip())
                    else:
                        print(f"Preskačem liniju {i+1}: ima {cols} kolona umesto {expected_cols}")
                
                # Napravi temp fajl
                temp_content = '\n'.join(clean_lines)
                from io import StringIO
                df = pd.read_csv(StringIO(temp_content), encoding='utf-8')
                
                print(f"Uspešno učitano {len(df)} čistih redova")
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
                
            except Exception as e3:
                print(f"Finalna greška pri učitavanju podataka: {e3}")
                return pd.DataFrame()
    
    except Exception as e:
        print(f"Neočekivana greška pri učitavanju podataka: {e}")
        return pd.DataFrame()

def create_matplotlib_chart(figure):
    """Konvertuj matplotlib figuru u base64 string za HTML"""
    img = BytesIO()
    figure.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()
    plt.close(figure)
    return img_b64

def generate_analytics_data():
    """Generiši sve analytics podatke i vizualizacije"""
    survey_df = load_survey_data()
    
    analytics = {
        'survey_charts': [],
        'summary_stats': {},
        'quiz_analysis': {}
    }
    
    if survey_df.empty:
        return analytics
    
    # === OSNOVNE STATISTIKE ===
    total_responses = len(survey_df)
    
    # Izračunaj employed_percentage
    employed_count = 0
    unemployed_count = 0
    if 'radni_odnos' in survey_df.columns:
        employed_count = (survey_df['radni_odnos'] == 'da').sum()
        unemployed_count = (survey_df['radni_odnos'] == 'ne').sum()
    employed_percentage = (employed_count / total_responses * 100) if total_responses > 0 else 0
    
    # Izračunaj koliko ljudi zna za ChatGPT i Copilot
    chatgpt_aware = 0
    copilot_aware = 0
    if 'poznati_ai_alati' in survey_df.columns:
        for tools in survey_df['poznati_ai_alati'].dropna():
            if 'ChatGPT' in str(tools):
                chatgpt_aware += 1
            if 'Copilot' in str(tools) or 'GitHub Copilot' in str(tools):
                copilot_aware += 1
    
    # Izračunaj prosečno AI korišćenje - koristi različite kolone za različite grupe
    avg_ai_usage = 0
    ai_usage_scores = []
    
    # Pokupi AI korišćenje iz različitih profesionalnih grupa
    for col in ['it_ai_koriscenje', 'prosveta_ai_koriscenje', 'medicina_ai_koriscenje', 
                'kreativna_ai_koriscenje', 'drustvene_ai_koriscenje', 'ostalo_ai_koriscenje']:
        if col in survey_df.columns:
            for value in survey_df[col].dropna():
                if value == 'da':
                    ai_usage_scores.append(5)  # Maksimalno korišćenje
                elif value == 'ne':
                    ai_usage_scores.append(1)  # Minimalno korišćenje
                elif value == 'ponekad':
                    ai_usage_scores.append(3)  # Umereno korišćenje
    
    if ai_usage_scores:
        avg_ai_usage = round(sum(ai_usage_scores) / len(ai_usage_scores), 2)
    
    # Izračunaj prosečno poznavanje programskih okruženja i jezika
    avg_programming_env = 0
    avg_programming_lang = 0
    
    # Ova polja nisu numerička u CSV-u, pa ćemo ih ostaviti na 0
    # ili dodati logiku za interpretaciju teksta ako je potrebno
    
    # Dodaj analizu IT industrije vs ostale grane
    it_better_ai_knowledge = 0
    it_users_count = 0
    non_it_users_count = 0
    it_ai_knowledge_avg = 0
    non_it_ai_knowledge_avg = 0
    
    if 'ciljana_grupa' in survey_df.columns:
        it_users = survey_df[survey_df['ciljana_grupa'] == 'it_industrija']
        non_it_users = survey_df[survey_df['ciljana_grupa'] != 'it_industrija']
        
        it_users_count = len(it_users)
        non_it_users_count = len(non_it_users)
        
        # Analiziraj koliko IT korisnici bolje poznaju AI alate
        it_chatgpt_aware = 0
        it_copilot_aware = 0
        non_it_chatgpt_aware = 0
        non_it_copilot_aware = 0
        
        # Broj IT korisnika koji poznaju ChatGPT i Copilot
        if 'poznati_ai_alati' in survey_df.columns:
            for tools in it_users['poznati_ai_alati'].dropna():
                if 'ChatGPT' in str(tools):
                    it_chatgpt_aware += 1
                if 'Copilot' in str(tools) or 'GitHub Copilot' in str(tools):
                    it_copilot_aware += 1
                    
            for tools in non_it_users['poznati_ai_alati'].dropna():
                if 'ChatGPT' in str(tools):
                    non_it_chatgpt_aware += 1
                if 'Copilot' in str(tools) or 'GitHub Copilot' in str(tools):
                    non_it_copilot_aware += 1
        
        # Izračunaj procenat poznavanja
        it_ai_knowledge_avg = ((it_chatgpt_aware + it_copilot_aware) / (it_users_count * 2) * 100) if it_users_count > 0 else 0
        non_it_ai_knowledge_avg = ((non_it_chatgpt_aware + non_it_copilot_aware) / (non_it_users_count * 2) * 100) if non_it_users_count > 0 else 0
        
        it_better_ai_knowledge = round(it_ai_knowledge_avg - non_it_ai_knowledge_avg, 1)
    
    analytics['summary_stats'] = {
        'total_responses': total_responses,
        'total_survey_responses': total_responses,  # Template očekuje ovaj ključ
        'employed_count': employed_count,
        'unemployed_count': unemployed_count,
        'avg_age': round(2025 - pd.to_numeric(survey_df['godina_rodjenja'], errors='coerce').mean(), 1) if 'godina_rodjenja' in survey_df.columns else 0,
        'countries': survey_df['drzava'].nunique() if 'drzava' in survey_df.columns else 0,
        'completion_rate': 100.0,  # Svi odgovori su kompletni
        'employed_percentage': round(employed_percentage, 1),
        'chatgpt_aware': chatgpt_aware,
        'copilot_aware': copilot_aware,
        'avg_programming_env': avg_programming_env,
        'avg_programming_lang': avg_programming_lang,
        'avg_ai_usage': avg_ai_usage,
        'it_better_ai_knowledge': it_better_ai_knowledge,
        'it_users_count': it_users_count,
        'non_it_users_count': non_it_users_count,
        'it_ai_knowledge_avg': round(it_ai_knowledge_avg, 1),
        'non_it_ai_knowledge_avg': round(non_it_ai_knowledge_avg, 1)
    }
    
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    # 1. DEMOGRAFSKI PODACI
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Godine rođenja
    if 'godina_rodjenja' in survey_df.columns:
        birth_years = survey_df['godina_rodjenja'].value_counts().sort_index()
        birth_years.plot(kind='bar', ax=ax1, color='#3498db', alpha=0.8)
        ax1.set_title('Distribucija Godine Rođenja', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Broj učesnika')
        ax1.tick_params(axis='x', rotation=45)
    
    # Države
    if 'drzava' in survey_df.columns:
        countries = survey_df['drzava'].value_counts().head(10)
        countries.plot(kind='barh', ax=ax2, color='#e74c3c', alpha=0.8)
        ax2.set_title('Top 10 Država', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Broj učesnika')
    
    # Stručna sprema
    if 'strucna_sprema' in survey_df.columns:
        education = survey_df['strucna_sprema'].value_counts()
        colors = ['#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        ax3.pie(education.values, labels=education.index, autopct='%1.1f%%', 
                colors=colors[:len(education)], startangle=90)
        ax3.set_title('Stručna Sprema', fontsize=12, fontweight='bold')
    
    # Radni status
    if 'radni_odnos' in survey_df.columns:
        work_status = survey_df['radni_odnos'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        ax4.pie(work_status.values, labels=['Zaposleni', 'Nezaposleni'], autopct='%1.1f%%',
                colors=colors, startangle=90)
        ax4.set_title('Status Zaposlenja', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    analytics['survey_charts'].append({
        'title': 'Demografski Profil Učesnika',
        'image': create_matplotlib_chart(fig)
    })
    
    # 📋 ANALIZA ANKETE O AI ALATA - POTPUNO NOVA SEKCIJA
    # Vizualizacije odgovora o korišćenju ChatGPT-a i Copilot-a među studentima
    
    # 1. OSNOVNA ANALIZA AI ALATA - Panoramski pregled
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('📋 Analiza Ankete o AI Alata - Osnovna Analiza', fontsize=16, fontweight='bold', y=0.98)
    
    try:
        # 1.1 ChatGPT vs Copilot Popularnost - Treemap Style Bar Chart
        if 'poznati_ai_alati' in survey_df.columns:
            chatgpt_users = survey_df['poznati_ai_alati'].astype(str).str.contains('ChatGPT', na=False).sum()
            copilot_users = survey_df['poznati_ai_alati'].astype(str).str.contains('GitHub Copilot', na=False).sum()
            claude_users = survey_df['poznati_ai_alati'].astype(str).str.contains('Claude', na=False).sum()
            dalle_users = survey_df['poznati_ai_alati'].astype(str).str.contains('DALL-E', na=False).sum()
            gemini_users = survey_df['poznati_ai_alati'].astype(str).str.contains('Gemini', na=False).sum()
            
            tools = ['ChatGPT', 'GitHub Copilot', 'Claude', 'DALL-E', 'Gemini']
            counts = [chatgpt_users, copilot_users, claude_users, dalle_users, gemini_users]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            
            bars = ax1.barh(tools, counts, color=colors, edgecolor='black', linewidth=1)
            ax1.set_xlabel('Broj Korisnika', fontweight='bold')
            ax1.set_title('🔥 TOP 5 AI Alata - Zastupljenost u Anketi', fontweight='bold', fontsize=12)
            ax1.grid(axis='x', alpha=0.3)
            
            # Dodaj vrednosti na barove
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax1.text(count + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{count} ({count/len(survey_df)*100:.1f}%)', 
                        va='center', fontweight='bold')
        
        # 1.2 Svrhe Korišćenja AI Alata - Word Cloud Style
        if 'svrhe_koriscenja' in survey_df.columns:
            purposes = []
            for purpose_text in survey_df['svrhe_koriscenja'].dropna():
                if isinstance(purpose_text, str) and purpose_text:
                    purposes.extend(purpose_text.split(','))
            
            # Brojanje najčešćih svrha
            purpose_counts = {}
            common_purposes = [
                'Pisanje teksta', 'Kreiranje slika', 'Učenje novih koncepata', 
                'Analiza podataka', 'Prevođenje jezika', 'Kreiranje prezentacija'
            ]
            
            for purpose in common_purposes:
                count = sum(1 for p in purposes if purpose.lower() in p.lower())
                if count > 0:
                    purpose_counts[purpose] = count
            
            if purpose_counts:
                purpose_labels = list(purpose_counts.keys())
                purpose_values = list(purpose_counts.values())
                colors_pie = plt.cm.Set3(np.linspace(0, 1, len(purpose_labels)))
                
                wedges, texts, autotexts = ax2.pie(purpose_values, labels=purpose_labels, autopct='%1.1f%%',
                                                  colors=colors_pie, startangle=90, textprops={'fontsize': 9})
                ax2.set_title('🎯 Svrhe Korišćenja AI Alata', fontweight='bold', fontsize=12)
                
                # Poboljšaj čitljivost
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
        
        # 1.3 AI Znanje vs Industrija - Heatmap sa statistikama
        if 'ciljana_grupa' in survey_df.columns and 'poznavanje_principa' in survey_df.columns:
            knowledge_levels = ['nimalo', 'malo', 'dosta', 'veoma_dosta']
            industries = ['it_industrija', 'prosveta', 'medicina', 'kreativna_filmska', 'drustvene_nauke']
            
            # Kreiraj matricu znanja po industrijama
            knowledge_matrix = np.zeros((len(industries), len(knowledge_levels)))
            
            for i, industry in enumerate(industries):
                industry_data = survey_df[survey_df['ciljana_grupa'] == industry]
                for j, level in enumerate(knowledge_levels):
                    count = (industry_data['poznavanje_principa'] == level).sum()
                    knowledge_matrix[i, j] = count
            
            # Normalizuj po redovima za bolje poređenje
            row_sums = knowledge_matrix.sum(axis=1)
            knowledge_matrix_norm = np.divide(knowledge_matrix, row_sums[:, np.newaxis], 
                                            out=np.zeros_like(knowledge_matrix), where=row_sums[:, np.newaxis]!=0) * 100
            
            im = ax3.imshow(knowledge_matrix_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=100)
            ax3.set_xticks(range(len(knowledge_levels)))
            ax3.set_yticks(range(len(industries)))
            ax3.set_xticklabels(['Nimalo', 'Malo', 'Dosta', 'Veoma Dosta'], rotation=45)
            ax3.set_yticklabels([ind.replace('_', ' ').title() for ind in industries])
            ax3.set_title('🧠 Nivo AI Znanja po Industrijama (%)', fontweight='bold', fontsize=12)
            
            # Dodaj brojeve u ćelije
            for i in range(len(industries)):
                for j in range(len(knowledge_levels)):
                    if row_sums[i] > 0:
                        text = ax3.text(j, i, f'{knowledge_matrix_norm[i, j]:.0f}%',
                                       ha="center", va="center", 
                                       color="white" if knowledge_matrix_norm[i, j] > 50 else "black",
                                       fontweight='bold')
            
            # Dodaj colorbar
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Procenat korisnika')
        
        # 1.4 Generacijska Analiza - Advanced Stacked Bar
        if 'godina_rodjenja' in survey_df.columns and 'poznati_ai_alati' in survey_df.columns:
            # Definiši generacije
            current_year = 2025
            survey_df['age'] = current_year - pd.to_numeric(survey_df['godina_rodjenja'], errors='coerce')
            
            def get_generation(age):
                if pd.isna(age):
                    return 'Nepoznato'
                elif age <= 25:
                    return 'Gen Z (≤25)'
                elif age <= 35:
                    return 'Mladi Milenijali (26-35)'
                elif age <= 45:
                    return 'Stariji Milenijali (36-45)'
                else:
                    return 'Gen X+ (46+)'
            
            survey_df['generation'] = survey_df['age'].apply(get_generation)
            generations = ['Gen Z (≤25)', 'Mladi Milenijali (26-35)', 'Stariji Milenijali (36-45)', 'Gen X+ (46+)']
            
            # Analiziraj korišćenje top AI alata po generacijama
            generation_data = {}
            for gen in generations:
                gen_data = survey_df[survey_df['generation'] == gen]
                if len(gen_data) > 0:
                    chatgpt = gen_data['poznati_ai_alati'].astype(str).str.contains('ChatGPT', na=False).sum()
                    copilot = gen_data['poznati_ai_alati'].astype(str).str.contains('GitHub Copilot', na=False).sum()
                    claude = gen_data['poznati_ai_alati'].astype(str).str.contains('Claude', na=False).sum()
                    dalle = gen_data['poznati_ai_alati'].astype(str).str.contains('DALL-E', na=False).sum()
                    
                    total = len(gen_data)
                    generation_data[gen] = {
                        'ChatGPT': (chatgpt/total)*100,
                        'Copilot': (copilot/total)*100,
                        'Claude': (claude/total)*100,
                        'DALL-E': (dalle/total)*100
                    }
            
            if generation_data:
                df_gen = pd.DataFrame(generation_data).T
                df_gen.plot(kind='bar', stacked=False, ax=ax4, 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                           width=0.8)
                ax4.set_title('🎭 AI Alati po Generacijama (%)', fontweight='bold', fontsize=12)
                ax4.set_xlabel('Generacija', fontweight='bold')
                ax4.set_ylabel('Procenat Korisnika', fontweight='bold')
                ax4.legend(title='AI Alati', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(axis='y', alpha=0.3)
    
    except Exception as e:
        print(f"Greška u osnovnoj AI analizi: {str(e)}")
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, f'Grafikon nedostupan\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    analytics['survey_charts'].append({
        'title': '📋 Analiza Ankete o AI Alata - Osnovna Analiza',
        'image': create_matplotlib_chart(fig)
    })
    
    # 2. NAPREDNA STATISTIČKA ANALIZA
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('📊 Napredna Statistička Analiza AI Korišćenja', fontsize=16, fontweight='bold', y=0.98)
    
    try:
        # 2.1 Korelacijska Matrica - AI Veštine
        correlation_fields = ['poznavanje_principa', 'prompt_engineering', 'tehnicko_razumevanje']
        if all(field in survey_df.columns for field in correlation_fields):
            # Mapiranje tekstualnih vrednosti na numeričke
            knowledge_mapping = {'nimalo': 1, 'malo': 2, 'dosta': 3, 'veoma_dosta': 4}
            prompt_mapping = {'ne': 1, 'da_teoretski': 2, 'da_praktikujem': 3}
            tech_mapping = {'Large Language Models (LLM)': 3, 'Fine-tuning modela na sopstvenim podacima': 4, 
                           'Rad sa embedding vektorima': 4, 'Ne razumem dovoljno': 1}
            
            corr_data = pd.DataFrame()
            corr_data['AI Znanje'] = survey_df['poznavanje_principa'].map(knowledge_mapping)
            corr_data['Prompt Eng.'] = survey_df['prompt_engineering'].map(prompt_mapping)
            
            # Za tehničko razumevanje, mapiranje prema složenosti
            tech_scores = []
            for tech in survey_df['tehnicko_razumevanje']:
                if isinstance(tech, str):
                    if 'Ne razumem' in tech:
                        tech_scores.append(1)
                    elif 'Large Language Models' in tech:
                        tech_scores.append(3)
                    elif any(advanced in tech for advanced in ['Fine-tuning', 'embedding', 'transformer']):
                        tech_scores.append(4)
                    else:
                        tech_scores.append(2)
                else:
                    tech_scores.append(np.nan)
            
            corr_data['Tehničko Razum.'] = tech_scores
            
            # Izračunaj korelacije
            correlation_matrix = corr_data.corr()
            
            # Kreiraj heatmap korelacije
            im = ax1.imshow(correlation_matrix.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax1.set_xticks(range(len(correlation_matrix.columns)))
            ax1.set_yticks(range(len(correlation_matrix.columns)))
            ax1.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax1.set_yticklabels(correlation_matrix.columns)
            ax1.set_title('🔗 Korelacijska Matrica AI Veština', fontweight='bold', fontsize=12)
            
            # Dodaj vrednosti korelacije
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    if not pd.isna(correlation_matrix.iloc[i, j]):
                        text = ax1.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", 
                                       color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
                                       fontweight='bold')
            
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Korelacija')
        
        # 2.2 Distribucija AI Znanja - Violin Plot sa statističkim podacima
        if 'poznavanje_principa' in survey_df.columns and 'ciljana_grupa' in survey_df.columns:
            knowledge_numeric = survey_df['poznavanje_principa'].map(knowledge_mapping)
            industries_clean = ['it_industrija', 'prosveta', 'medicina', 'kreativna_filmska', 'drustvene_nauke']
            
            plot_data = []
            labels = []
            stats_text = []
            
            for industry in industries_clean:
                industry_data = survey_df[survey_df['ciljana_grupa'] == industry]
                industry_knowledge = industry_data['poznavanje_principa'].map(knowledge_mapping).dropna()
                
                if len(industry_knowledge) > 0:
                    plot_data.append(industry_knowledge.values)
                    labels.append(industry.replace('_', ' ').title())
                    
                    # Izračunaj statistike
                    mean_val = industry_knowledge.mean()
                    median_val = industry_knowledge.median()
                    std_val = industry_knowledge.std()
                    stats_text.append(f'μ={mean_val:.2f}\nσ={std_val:.2f}')
            
            if plot_data:
                violin_parts = ax2.violinplot(plot_data, positions=range(len(plot_data)), 
                                            showmeans=True, showmedians=True)
                ax2.set_xticks(range(len(labels)))
                ax2.set_xticklabels(labels, rotation=45, ha='right')
                ax2.set_ylabel('AI Znanje Nivo (1-4)', fontweight='bold')
                ax2.set_title('📈 Distribucija AI Znanja po Sektorima', fontweight='bold', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                # Oboji violin plotove
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                for i, pc in enumerate(violin_parts['bodies']):
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)
                
                # Dodaj statistike kao tekst
                for i, stats in enumerate(stats_text):
                    ax2.text(i, max([max(data) for data in plot_data]) + 0.1, stats,
                            ha='center', va='bottom', fontsize=8, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 2.3 Trend Analiza - ChatGPT vs Copilot po Obrazovanju
        if 'strucna_sprema' in survey_df.columns and 'poznati_ai_alati' in survey_df.columns:
            education_levels = survey_df['strucna_sprema'].value_counts().index.tolist()
            
            chatgpt_by_edu = []
            copilot_by_edu = []
            education_labels = []
            
            for edu in education_levels:
                edu_data = survey_df[survey_df['strucna_sprema'] == edu]
                if len(edu_data) > 0:
                    chatgpt_pct = (edu_data['poznati_ai_alati'].astype(str).str.contains('ChatGPT', na=False).sum() / len(edu_data)) * 100
                    copilot_pct = (edu_data['poznati_ai_alati'].astype(str).str.contains('GitHub Copilot', na=False).sum() / len(edu_data)) * 100
                    
                    chatgpt_by_edu.append(chatgpt_pct)
                    copilot_by_edu.append(copilot_pct)
                    education_labels.append(edu)
            
            x = np.arange(len(education_labels))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, chatgpt_by_edu, width, label='ChatGPT', 
                           color='#FF6B6B', alpha=0.8, edgecolor='black')
            bars2 = ax3.bar(x + width/2, copilot_by_edu, width, label='GitHub Copilot', 
                           color='#4ECDC4', alpha=0.8, edgecolor='black')
            
            ax3.set_xlabel('Nivo Obrazovanja', fontweight='bold')
            ax3.set_ylabel('Procenat Korisnika', fontweight='bold')
            ax3.set_title('🎓 ChatGPT vs Copilot po Obrazovanju', fontweight='bold', fontsize=12)
            ax3.set_xticks(x)
            ax3.set_xticklabels(education_labels, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            # Dodaj vrednosti na barove
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 2.4 Radar Chart - Problemi i Izazovi AI
        if 'problemi_ai' in survey_df.columns:
            problem_categories = [
                'Pristrasnost u odgovorima', 'Generisanje lažnih informacija', 
                'Autorska prava i plagijati', 'Privatnost podataka', 
                'Tehnička ograničenja', 'Etičke dileme'
            ]
            
            problem_counts = []
            for problem in problem_categories:
                count = survey_df['problemi_ai'].astype(str).str.contains(problem, na=False).sum()
                problem_counts.append(count)
            
            # Normalizuj na procenat
            total_responses = len(survey_df['problemi_ai'].dropna())
            problem_percentages = [(count/total_responses)*100 for count in problem_counts]
            
            # Radar chart setup
            angles = np.linspace(0, 2 * np.pi, len(problem_categories), endpoint=False).tolist()
            problem_percentages += problem_percentages[:1]  # Complete the circle
            angles += angles[:1]
            
            ax4 = plt.subplot(224, projection='polar')
            ax4.plot(angles, problem_percentages, 'o-', linewidth=2, color='#FF6B6B')
            ax4.fill(angles, problem_percentages, alpha=0.25, color='#FF6B6B')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels([cat.replace(' ', '\n') for cat in problem_categories], fontsize=9)
            ax4.set_ylim(0, max(problem_percentages) * 1.2)
            ax4.set_title('⚠️ Identifikovani Problemi AI (%)', fontweight='bold', fontsize=12, pad=20)
            ax4.grid(True)
            
            # Dodaj vrednosti
            for angle, pct in zip(angles[:-1], problem_percentages[:-1]):
                ax4.text(angle, pct + max(problem_percentages)*0.05, f'{pct:.1f}%', 
                        ha='center', va='center', fontweight='bold', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    except Exception as e:
        print(f"Greška u naprednoj AI analizi: {str(e)}")
        for ax in [ax1, ax2, ax3]:
            if ax != ax4:  # ax4 je polar, drugačije se rukuje
                ax.text(0.5, 0.5, f'Grafikon nedostupan\n{str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    analytics['survey_charts'].append({
        'title': '📊 Napredna Statistička Analiza AI Korišćenja',
        'image': create_matplotlib_chart(fig)
    })
    
    # 3. ZNAČAJNI STATISTIČKI PODACI ZA BUDUĆA ISTRAŽIVANJA
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('🔬 Ključni Nalazi za Buduća Istraživanja', fontsize=16, fontweight='bold', y=0.98)
    
    try:
        # 3.1 AI Adoptacija Indeks - Kompozitni pokazatelj
        if all(field in survey_df.columns for field in ['poznati_ai_alati', 'svrhe_koriscenja', 'poznavanje_principa']):
            
            # Izračunaj AI Adoptacijski Indeks (0-100)
            ai_adoption_scores = []
            industries_for_adoption = []
            
            for idx, row in survey_df.iterrows():
                score = 0
                
                # Broj poznatih AI alata (max 30 poena)
                tools = str(row.get('poznati_ai_alati', ''))
                tool_count = sum(1 for tool in ['ChatGPT', 'GitHub Copilot', 'Claude', 'DALL-E', 'Gemini'] if tool in tools)
                score += min(tool_count * 6, 30)
                
                # Raznovrsnost svrha (max 25 poena)
                purposes = str(row.get('svrhe_koriscenja', ''))
                purpose_count = len([p for p in purposes.split(',') if p.strip()]) if purposes != 'nan' else 0
                score += min(purpose_count * 5, 25)
                
                # Nivo znanja (max 30 poena)
                knowledge = row.get('poznavanje_principa', '')
                knowledge_score = {'nimalo': 0, 'malo': 10, 'dosta': 20, 'veoma_dosta': 30}.get(knowledge, 0)
                score += knowledge_score
                
                # Prompt engineering (max 15 poena)
                prompt_skill = row.get('prompt_engineering', '')
                prompt_score = {'ne': 0, 'da_teoretski': 7, 'da_praktikujem': 15}.get(prompt_skill, 0)
                score += prompt_score
                
                ai_adoption_scores.append(score)
                industries_for_adoption.append(row.get('ciljana_grupa', 'nepoznato'))
            
            # Grupiši po industrijama
            adoption_by_industry = {}
            industries_clean = ['it_industrija', 'prosveta', 'medicina', 'kreativna_filmska', 'drustvene_nauke']
            
            for industry in industries_clean:
                industry_scores = [score for score, ind in zip(ai_adoption_scores, industries_for_adoption) if ind == industry]
                if industry_scores:
                    avg_score = np.mean(industry_scores)
                    adoption_by_industry[industry.replace('_', ' ').title()] = avg_score
            
            if adoption_by_industry:
                industries = list(adoption_by_industry.keys())
                scores = list(adoption_by_industry.values())
                colors = plt.cm.RdYlGn(np.array(scores) / 100)
                
                bars = ax1.barh(industries, scores, color=colors, edgecolor='black', linewidth=1)
                ax1.set_xlabel('AI Adoptacijski Indeks (0-100)', fontweight='bold')
                ax1.set_title('🚀 AI Adoptacijski Indeks po Sektorima', fontweight='bold', fontsize=12)
                ax1.grid(axis='x', alpha=0.3)
                
                # Dodaj vrednosti i kategorije
                for bar, score in zip(bars, scores):
                    category = "Visok" if score >= 70 else "Srednji" if score >= 40 else "Nizak"
                    ax1.text(score + 2, bar.get_y() + bar.get_height()/2, 
                            f'{score:.1f} ({category})', va='center', fontweight='bold')
        
        # 3.2 Prediktivni Model - AI Znanje vs Godina
        if 'godina_rodjenja' in survey_df.columns and 'poznavanje_principa' in survey_df.columns:
            birth_years = pd.to_numeric(survey_df['godina_rodjenja'], errors='coerce')
            knowledge_numeric = survey_df['poznavanje_principa'].map({'nimalo': 1, 'malo': 2, 'dosta': 3, 'veoma_dosta': 4})
            
            # Filtriraj validne podatke
            valid_data = pd.DataFrame({
                'birth_year': birth_years,
                'knowledge': knowledge_numeric
            }).dropna()
            
            if len(valid_data) > 10:
                # Scatter plot sa trendom
                scatter = ax2.scatter(valid_data['birth_year'], valid_data['knowledge'], 
                                    alpha=0.6, c=valid_data['knowledge'], cmap='viridis', 
                                    s=60, edgecolors='black', linewidth=0.5)
                
                # Dodaj trend liniju
                z = np.polyfit(valid_data['birth_year'], valid_data['knowledge'], 1)
                p = np.poly1d(z)
                trend_x = np.linspace(valid_data['birth_year'].min(), valid_data['birth_year'].max(), 100)
                ax2.plot(trend_x, p(trend_x), "r--", alpha=0.8, linewidth=3, label=f'Trend: y={z[0]:.3f}x+{z[1]:.1f}')
                
                ax2.set_xlabel('Godina Rođenja', fontweight='bold')
                ax2.set_ylabel('AI Znanje Nivo (1-4)', fontweight='bold')
                ax2.set_title('📈 Prediktivni Trend: AI Znanje kroz Generacije', fontweight='bold', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Dodaj projekciju
                future_years = [2030, 2035, 2040]
                future_knowledge = [p(year) for year in future_years]
                ax2.plot(future_years, future_knowledge, 'ro', markersize=8, label='Projekcija')
                
                # Statistike
                correlation = valid_data['birth_year'].corr(valid_data['knowledge'])
                r_squared = correlation ** 2
                ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}\nKorelacija = {correlation:.3f}', 
                        transform=ax2.transAxes, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        # 3.3 Kompetencijski Gap - Trenutno vs Potrebno Znanje
        if 'poznavanje_principa' in survey_df.columns and 'tehnicko_razumevanje' in survey_df.columns:
            
            # Mapiranje trenutnog znanja
            current_knowledge = survey_df['poznavanje_principa'].map({'nimalo': 1, 'malo': 2, 'dosta': 3, 'veoma_dosta': 4})
            
            # Mapiranje potrebnog znanja na osnovu tehničkog razumevanja
            needed_knowledge = []
            for tech in survey_df['tehnicko_razumevanje']:
                if isinstance(tech, str):
                    if 'Ne razumem' in tech:
                        needed_knowledge.append(2)  # Potrebno osnovno znanje
                    elif 'Large Language Models' in tech:
                        needed_knowledge.append(3)  # Potrebno solidno znanje
                    elif any(advanced in tech for advanced in ['Fine-tuning', 'embedding']):
                        needed_knowledge.append(4)  # Potrebno napredno znanje
                    else:
                        needed_knowledge.append(2.5)
                else:
                    needed_knowledge.append(np.nan)
            
            # Izračunaj gap
            gap_data = pd.DataFrame({
                'current': current_knowledge,
                'needed': needed_knowledge,
                'industry': survey_df['ciljana_grupa']
            }).dropna()
            
            gap_data['gap'] = gap_data['needed'] - gap_data['current']
            
            # Grupiši po industrijama
            gap_by_industry = {}
            for industry in industries_clean:
                industry_gaps = gap_data[gap_data['industry'] == industry]['gap']
                if len(industry_gaps) > 0:
                    avg_gap = industry_gaps.mean()
                    gap_by_industry[industry.replace('_', ' ').title()] = avg_gap
            
            if gap_by_industry:
                industries = list(gap_by_industry.keys())
                gaps = list(gap_by_industry.values())
                
                # Pozitivan gap = nedostatak znanja, negativan = višak znanja
                colors = ['#FF6B6B' if gap > 0 else '#4ECDC4' for gap in gaps]
                
                bars = ax3.barh(industries, gaps, color=colors, edgecolor='black', linewidth=1)
                ax3.set_xlabel('Kompetencijski Gap (Potrebno - Trenutno)', fontweight='bold')
                ax3.set_title('⚖️ Kompetencijski Gap po Sektorima', fontweight='bold', fontsize=12)
                ax3.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                ax3.grid(axis='x', alpha=0.3)
                
                # Dodaj vrednosti i interpretacije
                for bar, gap in zip(bars, gaps):
                    interpretation = "Nedostatak" if gap > 0.5 else "Balans" if abs(gap) <= 0.5 else "Višak"
                    color = 'white' if abs(gap) > 0.5 else 'black'
                    ax3.text(gap + (0.1 if gap >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                            f'{gap:.2f}\n({interpretation})', ha='left' if gap >= 0 else 'right', 
                            va='center', fontweight='bold', color=color)
        
        # 3.4 Budući Trendovi - Inovacije i Preferencije
        innovation_fields = ['chatgpt_omni', 'copilot_workspace', 'gpt_realtime', 'anthropic_model']
        if all(field in survey_df.columns for field in innovation_fields):
            
            # Mapa savremenih AI funkcionalnosti
            innovation_map = {
                'chatgpt_omni': 'Multimodalni AI (GPT-4)',
                'copilot_workspace': 'AI Development Environment',
                'gpt_realtime': 'Realtime AI Komunikacija',
                'anthropic_model': 'Etička AI (Claude)'
            }
            
            # Analiza svesti o inovacijama
            innovation_awareness = {}
            for field, label in innovation_map.items():
                if field in survey_df.columns:
                    # Broj ljudi koji je dao bilo kakav odgovor (ne prazan)
                    aware_count = len(survey_df[survey_df[field].notna() & (survey_df[field] != '')])
                    total_count = len(survey_df)
                    awareness_pct = (aware_count / total_count) * 100
                    innovation_awareness[label] = awareness_pct
            
            if innovation_awareness:
                # Kreiraj polumesec grafikon (donut)
                labels = list(innovation_awareness.keys())
                sizes = list(innovation_awareness.values())
                colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700']
                
                # Napravi eksploziju za najvažnije inovacije
                explode = [0.1 if size > np.mean(sizes) else 0 for size in sizes]
                
                wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                  colors=colors, startangle=90, explode=explode,
                                                  wedgeprops=dict(width=0.6, edgecolor='black'))
                ax4.set_title('🔮 Svest o AI Inovacijama (% ispitanika)', fontweight='bold', fontsize=12)
                
                # Dodaj centralnu statistiku
                avg_awareness = np.mean(sizes)
                ax4.text(0, 0, f'Prosečna\nSvest\n{avg_awareness:.1f}%', 
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))
                
                # Poboljšaj čitljivost
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                # Dodaj interpretaciju ispod
                interpretation = "Visoka" if avg_awareness > 70 else "Srednja" if avg_awareness > 40 else "Niska"
                ax4.text(0, -1.3, f'Opšta svest o AI inovacijama: {interpretation}', 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))
    
    except Exception as e:
        print(f"Greška u analizi za buduća istraživanja: {str(e)}")
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, f'Grafikon nedostupan\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    analytics['survey_charts'].append({
        'title': '🔬 Ključni Nalazi za Buduća Istraživanja',
        'image': create_matplotlib_chart(fig)
    })
    
    # 4. Izračunaj ključne statistike za prikazivanje
    analytics['key_insights'] = {}
    
    try:
        # Osnovna demografija
        total_responses = len(survey_df)
        analytics['key_insights']['total_responses'] = total_responses
        
        # ChatGPT vs Copilot statistike
        if 'poznati_ai_alati' in survey_df.columns:
            chatgpt_users = survey_df['poznati_ai_alati'].astype(str).str.contains('ChatGPT', na=False).sum()
            copilot_users = survey_df['poznati_ai_alati'].astype(str).str.contains('GitHub Copilot', na=False).sum()
            
            analytics['key_insights']['chatgpt_percentage'] = (chatgpt_users / total_responses) * 100
            analytics['key_insights']['copilot_percentage'] = (copilot_users / total_responses) * 100
            analytics['key_insights']['both_tools_users'] = survey_df['poznati_ai_alati'].astype(str).str.contains('ChatGPT.*GitHub Copilot|GitHub Copilot.*ChatGPT', na=False).sum()
        
        # Distribucija po industrijama
        if 'ciljana_grupa' in survey_df.columns:
            industry_counts = survey_df['ciljana_grupa'].value_counts()
            analytics['key_insights']['dominant_industry'] = industry_counts.index[0] if len(industry_counts) > 0 else 'N/A'
            analytics['key_insights']['dominant_industry_pct'] = (industry_counts.iloc[0] / total_responses) * 100 if len(industry_counts) > 0 else 0
        
        # AI znanje statistike
        if 'poznavanje_principa' in survey_df.columns:
            knowledge_counts = survey_df['poznavanje_principa'].value_counts()
            advanced_users = knowledge_counts.get('veoma_dosta', 0) + knowledge_counts.get('dosta', 0)
            analytics['key_insights']['advanced_users_pct'] = (advanced_users / total_responses) * 100
        
        # Generacijske statistike
        if 'godina_rodjenja' in survey_df.columns:
            birth_years = pd.to_numeric(survey_df['godina_rodjenja'], errors='coerce').dropna()
            current_year = 2025
            ages = current_year - birth_years
            gen_z_count = (ages <= 25).sum()
            millennials_count = ((ages > 25) & (ages <= 40)).sum()
            
            analytics['key_insights']['gen_z_pct'] = (gen_z_count / len(ages)) * 100
            analytics['key_insights']['millennials_pct'] = (millennials_count / len(ages)) * 100
            analytics['key_insights']['avg_age'] = ages.mean()
        
        # Problemi i izazovi
        if 'problemi_ai' in survey_df.columns:
            problem_text = ' '.join(survey_df['problemi_ai'].dropna().astype(str))
            bias_mentions = problem_text.lower().count('pristrasnost')
            fake_info_mentions = problem_text.lower().count('lažnih')
            copyright_mentions = problem_text.lower().count('autorska')
            
            total_problem_responses = len(survey_df['problemi_ai'].dropna())
            if total_problem_responses > 0:
                analytics['key_insights']['main_concern'] = 'Pristrasnost' if bias_mentions >= fake_info_mentions and bias_mentions >= copyright_mentions else 'Lažne informacije' if fake_info_mentions >= copyright_mentions else 'Autorska prava'
                analytics['key_insights']['concern_pct'] = max(bias_mentions, fake_info_mentions, copyright_mentions) / total_problem_responses * 100
        
        # Edukacija i rad
        if 'strucna_sprema' in survey_df.columns:
            education_counts = survey_df['strucna_sprema'].value_counts()
            higher_ed = education_counts.get('master', 0) + education_counts.get('doktorat', 0)
            analytics['key_insights']['higher_education_pct'] = (higher_ed / total_responses) * 100
        
        if 'radni_odnos' in survey_df.columns:
            employed = survey_df['radni_odnos'].value_counts().get('da', 0)
            analytics['key_insights']['employment_rate'] = (employed / total_responses) * 100
    
    except Exception as e:
        print(f"Greška u izračunu ključnih statistika: {str(e)}")
        analytics['key_insights'] = {
            'total_responses': len(survey_df),
            'error': str(e)
        }
    
    # 3. KVIZ ANALIZA
    quiz_fields = [
        'chatgpt_omni', 'copilot_task', 'copilot_chat', 'google_model',
        'gpt_realtime', 'codex_successor', 'chatgpt_data_analysis', 'copilot_workspace',
        'anthropic_model', 'creativity_parameter', 'transformer_basis', 'university_guidelines'
    ]
    
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
    
    quiz_results = {}
    quiz_scores = []
    
    for field in quiz_fields:
        if field in survey_df.columns:
            correct = (survey_df[field] == correct_answers[field]).sum()
            total = len(survey_df[field].dropna())
            if total > 0:
                accuracy = (correct / total) * 100
                quiz_results[field] = {
                    'correct': correct,
                    'total': total,
                    'accuracy': accuracy
                }
    
    # Izračunaj individualne skorove
    for _, row in survey_df.iterrows():
        score = 0
        answered = 0
        for field in quiz_fields:
            if field in row and pd.notna(row[field]) and row[field]:
                answered += 1
                if row[field] == correct_answers[field]:
                    score += 1
        if answered > 0:
            quiz_scores.append((score / answered) * 100)
    
    # Dodaj score raspodelu
    excellent_count = sum(1 for score in quiz_scores if score >= 80)
    good_count = sum(1 for score in quiz_scores if 60 <= score < 80)
    average_count = sum(1 for score in quiz_scores if 40 <= score < 60)
    poor_count = sum(1 for score in quiz_scores if score < 40)
    
    analytics['quiz_analysis'] = {
        'results': quiz_results,
        'avg_score': sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0,
        'scores': quiz_scores,
        'excellent_count': excellent_count,
        'good_count': good_count,
        'average_count': average_count,
        'poor_count': poor_count
    }
    
    # Kreiraj grafikon kviz rezultata
    if quiz_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Tačnost po pitanjima
        questions = list(quiz_results.keys())
        accuracies = [quiz_results[q]['accuracy'] for q in questions]
        
        # Skrati nazive pitanja za bolje prikazivanje
        short_names = [q.replace('_', ' ').title()[:15] + '...' for q in questions]
        
        ax1.barh(short_names, accuracies, color='#3498db', alpha=0.8)
        ax1.set_title('Tačnost Odgovora po Kviz Pitanjima', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Procenat tačnih odgovora')
        
        # Distribucija skorova
        if quiz_scores:
            ax2.hist(quiz_scores, bins=10, color='#2ecc71', alpha=0.8, edgecolor='black')
            ax2.set_title('Distribucija Kviz Skorova', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Procenat tačnih odgovora')
            ax2.set_ylabel('Broj učesnika')
            ax2.axvline(sum(quiz_scores)/len(quiz_scores), color='red', linestyle='--', 
                       label=f'Prosek: {sum(quiz_scores)/len(quiz_scores):.1f}%')
            ax2.legend()
        
        plt.tight_layout()
        analytics['survey_charts'].append({
            'title': 'Analiza Kviz Performansi',
            'image': create_matplotlib_chart(fig)
        })
    
    # 4. TEHNIČKO RAZUMEVANJE
    if 'transformer_elementi' in survey_df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Transformer elementi poznavanje
        all_elements = []
        for elements in survey_df['transformer_elementi'].dropna():
            if elements:
                all_elements.extend([elem.strip() for elem in elements.split(',')])
        
        if all_elements:
            element_counts = pd.Series(all_elements).value_counts().head(8)
            element_counts.plot(kind='barh', ax=ax1, color='#9b59b6', alpha=0.8)
            ax1.set_title('Poznavanje Transformer Elemenata', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Broj spomenuta')
        
        # Tehnike treniranja
        if 'tehnike_treniranja' in survey_df.columns:
            all_techniques = []
            for techniques in survey_df['tehnike_treniranja'].dropna():
                if techniques:
                    all_techniques.extend([tech.strip() for tech in techniques.split(',')])
            
            if all_techniques:
                tech_counts = pd.Series(all_techniques).value_counts().head(8)
                tech_counts.plot(kind='bar', ax=ax2, color='#34495e', alpha=0.8)
                ax2.set_title('Poznavanje Tehnika Treniranja', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Broj spomenuta')
                ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        analytics['survey_charts'].append({
            'title': 'Tehničko Razumevanje AI',
            'image': create_matplotlib_chart(fig)
        })
    
    return analytics

def create_matplotlib_chart(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    import io
    import base64
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_string = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)
    return img_string

def init_csv_file():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp',
                'godina_rodjenja',
                'drzava',
                'strucna_sprema',
                'radni_odnos',
                'ciljana_grupa',
                'godine_staza',
                'institucija',
                # IT Industrija fields
                'it_pozicije',
                'it_tehnologije',
                'it_ai_koriscenje',
                # Prosveta fields
                'prosveta_nivo',
                'prosveta_predmeti',
                'prosveta_ai_koriscenje',
                # Medicina fields
                'medicina_oblast',
                'medicina_pozicije',
                'medicina_ai_koriscenje',
                # Kreativna industrija fields
                'kreativna_oblasti',
                'kreativna_ai_koriscenje',
                # Društvene nauke fields
                'drustvene_oblast',
                'drustvene_aktivnosti',
                'drustvene_ai_koriscenje',
                # Ostalo fields
                'ostalo_oblast',
                'ostalo_ai_koriscenje',
                'ostalo_ekspertiza',
                # Rest of existing fields
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
            ])

@app.route('/')
def index():
    """Home page with survey form"""
    return render_template('survey.html')

@app.route('/submit', methods=['POST'])
def submit_survey():
    """Handle survey form submission"""
    try:
        # Get form data
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'godina_rodjenja': request.form.get('godina_rodjenja', ''),
            'drzava': request.form.get('drzava', ''),
            'strucna_sprema': request.form.get('strucna_sprema', ''),
            'radni_odnos': request.form.get('radni_odnos', ''),
            'ciljana_grupa': request.form.get('ciljana_grupa', ''),
            'godine_staza': request.form.get('godine_staza', ''),
            'institucija': request.form.get('institucija', ''),
            # IT Industrija fields
            'it_pozicije': ','.join(request.form.getlist('it_pozicije')),
            'it_tehnologije': ','.join(request.form.getlist('it_tehnologije')),
            'it_ai_koriscenje': request.form.get('it_ai_koriscenje', ''),
            # Prosveta fields
            'prosveta_nivo': ','.join(request.form.getlist('prosveta_nivo')),
            'prosveta_predmeti': request.form.get('prosveta_predmeti', ''),
            'prosveta_ai_koriscenje': request.form.get('prosveta_ai_koriscenje', ''),
            # Medicina fields
            'medicina_oblast': request.form.get('medicina_oblast', ''),
            'medicina_pozicije': ','.join(request.form.getlist('medicina_pozicije')),
            'medicina_ai_koriscenje': request.form.get('medicina_ai_koriscenje', ''),
            # Kreativna industrija fields
            'kreativna_oblasti': ','.join(request.form.getlist('kreativna_oblasti')),
            'kreativna_ai_koriscenje': request.form.get('kreativna_ai_koriscenje', ''),
            # Društvene nauke fields
            'drustvene_oblast': request.form.get('drustvene_oblast', ''),
            'drustvene_aktivnosti': ','.join(request.form.getlist('drustvene_aktivnosti')),
            'drustvene_ai_koriscenje': request.form.get('drustvene_ai_koriscenje', ''),
            # Ostalo fields
            'ostalo_oblast': request.form.get('ostalo_oblast', ''),
            'ostalo_ai_koriscenje': request.form.get('ostalo_ai_koriscenje', ''),
            'ostalo_ekspertiza': request.form.get('ostalo_ekspertiza', ''),
            # Rest of the existing fields
            'poznati_ai_alati': ','.join(request.form.getlist('poznati_ai_alati')),
            'ostalo_poznati_ai_alati': request.form.get('ostalo_poznati_ai_alati', ''),
            'svrhe_koriscenja': ','.join(request.form.getlist('svrhe_koriscenja')),
            'ostalo_svrhe': request.form.get('ostalo_svrhe', ''),
            'poznavanje_principa': request.form.get('poznavanje_principa', ''),
            'prompt_engineering': request.form.get('prompt_engineering', ''),
            'tehnicko_razumevanje': ','.join(request.form.getlist('tehnicko_razumevanje')),
            'problemi_ai': ','.join(request.form.getlist('problemi_ai')),
            'pravni_okviri': ','.join(request.form.getlist('pravni_okviri')),
            'provere_ai': ','.join(request.form.getlist('provere_ai')),
            'metode_evaluacije': ','.join(request.form.getlist('metode_evaluacije')),
            'ogranicenja_ai': request.form.get('ogranicenja_ai', ''),
            'transformer_elementi': ','.join(request.form.getlist('transformer_elementi')),
            'tehnike_treniranja': ','.join(request.form.getlist('tehnike_treniranja')),
            'koncepti_poznavanje': request.form.get('koncepti_poznavanje', ''),
            # Quiz fields
            'chatgpt_omni': request.form.get('chatgpt_omni', ''),
            'copilot_task': request.form.get('copilot_task', ''),
            'copilot_chat': request.form.get('copilot_chat', ''),
            'google_model': request.form.get('google_model', ''),
            'gpt_realtime': request.form.get('gpt_realtime', ''),
            'codex_successor': request.form.get('codex_successor', ''),
            'chatgpt_data_analysis': request.form.get('chatgpt_data_analysis', ''),
            'copilot_workspace': request.form.get('copilot_workspace', ''),
            'anthropic_model': request.form.get('anthropic_model', ''),
            'creativity_parameter': request.form.get('creativity_parameter', ''),
            'transformer_basis': request.form.get('transformer_basis', ''),
            'university_guidelines': request.form.get('university_guidelines', '')
        }
        
        # Save to CSV
        with open(RESPONSES_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                data['timestamp'],
                data['godina_rodjenja'],
                data['drzava'],
                data['strucna_sprema'],
                data['radni_odnos'],
                data['ciljana_grupa'],
                data['godine_staza'],
                data['institucija'],
                # IT Industrija fields
                data['it_pozicije'],
                data['it_tehnologije'],
                data['it_ai_koriscenje'],
                # Prosveta fields
                data['prosveta_nivo'],
                data['prosveta_predmeti'],
                data['prosveta_ai_koriscenje'],
                # Medicina fields
                data['medicina_oblast'],
                data['medicina_pozicije'],
                data['medicina_ai_koriscenje'],
                # Kreativna industrija fields
                data['kreativna_oblasti'],
                data['kreativna_ai_koriscenje'],
                # Društvene nauke fields
                data['drustvene_oblast'],
                data['drustvene_aktivnosti'],
                data['drustvene_ai_koriscenje'],
                # Ostalo fields
                data['ostalo_oblast'],
                data['ostalo_ai_koriscenje'],
                data['ostalo_ekspertiza'],
                # Rest of existing fields
                data['poznati_ai_alati'],
                data['ostalo_poznati_ai_alati'],
                data['svrhe_koriscenja'],
                data['ostalo_svrhe'],
                data['poznavanje_principa'],
                data['prompt_engineering'],
                data['tehnicko_razumevanje'],
                data['problemi_ai'],
                data['pravni_okviri'],
                data['provere_ai'],
                data['metode_evaluacije'],
                data['ogranicenja_ai'],
                data['transformer_elementi'],
                data['tehnike_treniranja'],
                data['koncepti_poznavanje'],
                data['chatgpt_omni'],
                data['copilot_task'],
                data['copilot_chat'],
                data['google_model'],
                data['gpt_realtime'],
                data['codex_successor'],
                data['chatgpt_data_analysis'],
                data['copilot_workspace'],
                data['anthropic_model'],
                data['creativity_parameter'],
                data['transformer_basis'],
                data['university_guidelines']
            ])
        
        # Data Science Extension Integration
        if DATA_SCIENCE_AVAILABLE:
            try:
                data_manager = DataScienceManager()
                participant_id = data_manager.process_survey_response(data)
                session['participant_id'] = participant_id
                session['data_science_enabled'] = True
                flash(f'Podaci uspešno sačuvani! Participant ID: {participant_id}', 'success')
            except Exception as ds_error:
                print(f"Data Science processing failed: {ds_error}")
                flash('Osnovni podaci sačuvani, napredna analiza nedostupna.', 'warning')
        else:
            flash('Hvala vam! Vaš odgovor je uspešno zabeležen.', 'success')
        
        return redirect(url_for('success'))
        
    except Exception as e:
        flash(f'Došlo je do greške: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/success')
def success():
    """Success page after survey submission"""
    return render_template('success.html')

@app.route('/results')
def results():
    """View survey results and basic statistics"""
    try:
        responses = []
        if os.path.exists(RESPONSES_FILE):
            with open(RESPONSES_FILE, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                responses = list(reader)
        
        print(f"DEBUG: Učitano {len(responses)} odgovora iz CSV-a")
        
        # Basic statistics sa ispravnim poljima
        total_responses = len(responses)
        
        # Brojimo zaposlene vs nezaposlene
        employed_count = sum(1 for r in responses if r.get('radni_odnos') == 'da')
        
        # Brojimo poznavanje AI alata
        ai_tools_users = sum(1 for r in responses if r.get('poznati_ai_alati', ''))
        
        # Prosečno znanje AI alata
        ai_knowledge_scores = []
        for r in responses:
            score = r.get('generativni_ai_poznavanje', '')
            if score and score.isdigit():
                ai_knowledge_scores.append(int(score))
        
        avg_ai_knowledge = round(sum(ai_knowledge_scores) / len(ai_knowledge_scores), 2) if ai_knowledge_scores else 0
        
        # Brojimo kviz rezultate
        quiz_fields = [
            'chatgpt_omni', 'copilot_task', 'copilot_chat', 'google_model',
            'gpt_realtime', 'codex_successor', 'chatgpt_data_analysis', 'copilot_workspace',
            'anthropic_model', 'creativity_parameter', 'transformer_basis', 'university_guidelines'
        ]
        
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
        
        quiz_scores = []
        for response in responses:
            correct_count = 0
            total_answered = 0
            for field in quiz_fields:
                answer = response.get(field, '')
                if answer:
                    total_answered += 1
                    if answer == correct_answers[field]:
                        correct_count += 1
            if total_answered > 0:
                quiz_scores.append(round((correct_count / total_answered) * 100, 1))
        
        avg_quiz_score = round(sum(quiz_scores) / len(quiz_scores), 1) if quiz_scores else 0
        
        stats = {
            'total_responses': total_responses,
            'employed_count': employed_count,
            'employed_percentage': round((employed_count / total_responses * 100) if total_responses > 0 else 0, 1),
            'ai_tools_users': ai_tools_users,
            'ai_tools_percentage': round((ai_tools_users / total_responses * 100) if total_responses > 0 else 0, 1),
            'avg_ai_knowledge': avg_ai_knowledge,
            'avg_quiz_score': avg_quiz_score,
            'quiz_participants': len(quiz_scores)
        }
        
        return render_template('results.html', responses=responses, stats=stats)
        
    except Exception as e:
        flash(f'Greška pri učitavanju rezultata: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/analytics')
def analytics():
    """Napredna analitika i vizualizacije podataka"""
    try:
        analytics_data = generate_analytics_data()
        return render_template('analytics.html', analytics=analytics_data)
    except Exception as e:
        flash(f'Greška pri generisanju analitike: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/data_science')
def data_science_dashboard():
    """Napredna Data Science analiza dashboard"""
    if not DATA_SCIENCE_AVAILABLE:
        flash('Data Science funkcionalnost nije dostupna. Instalirajte potrebne biblioteke.', 'error')
        return redirect(url_for('analytics'))
    
    try:
        # Kreiraj data manager instance
        data_manager = DataScienceManager()
        advanced_analytics = AdvancedAnalytics()
        
        # PROVERI I UČITAJ POSTOJEĆE PODATKE
        # Ako nema data science fajlova, ali ima survey_responses.csv, obradi ih
        if not os.path.exists('data/participants.csv') and os.path.exists(RESPONSES_FILE):
            print("🔄 Učitavam postojeće survey podatke...")
            data_manager.process_existing_survey_data(RESPONSES_FILE)
        
        # Dobij osnovni summary
        summary = data_manager.get_analytics_summary()
        
        # Proverava da li postoje podaci
        if summary.get('total_participants', 0) == 0:
            # Pokušaj ponovo sa učitavanjem podataka
            if os.path.exists(RESPONSES_FILE):
                print("🔄 Ponovni pokušaj učitavanja postojećih podataka...")
                data_manager.process_existing_survey_data(RESPONSES_FILE)
                summary = data_manager.get_analytics_summary()
                
                if summary.get('total_participants', 0) == 0:
                    flash('Nema dovoljno podataka za naprednu analizu. Potrebno je prvo da se pošalju odgovori.', 'warning')
                    return redirect(url_for('analytics'))
            else:
                flash('Nema dovoljno podataka za naprednu analizu. Potrebno je prvo da se pošalju odgovori.', 'warning')
                return redirect(url_for('analytics'))
        
        # Generiši NetworkX graf
        try:
            G = data_manager.create_network_graph()
            
            # Osnovne statistike
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            # Napredne network analize
            advanced_stats = {}
            
            if num_nodes > 0:
                # Average Degree
                degrees = dict(G.degree())
                avg_degree = (sum(degrees.values()) / len(degrees))/2 if degrees else 0
                
                # Average Weighted Degree
                if G.edges(data=True):
                    weighted_degrees = dict(G.degree(weight='weight'))
                    avg_weighted_degree = (sum(weighted_degrees.values()) / len(weighted_degrees))/2 if weighted_degrees else 0
                else:
                    avg_weighted_degree = 0
                
                # Network Diameter (samo za povezane grafove)
                try:
                    if nx.is_connected(G):
                        diameter = nx.diameter(G)
                    else:
                        # Za nepovezane grafove, uzmi najveći dijametar komponenti
                        components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
                        diameters = []
                        for comp in components:
                            if len(comp) > 1:
                                diameters.append(nx.diameter(comp))
                        diameter = max(diameters) if diameters else 0
                except:
                    diameter = "N/A"
                
                # Clustering Coefficient
                try:
                    clustering_coeff = nx.average_clustering(G)
                except:
                    clustering_coeff = 0
                
                # PageRank (top 3)
                try:
                    pagerank = nx.pagerank(G)
                    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
                    pagerank_top = [{"node": node, "score": round(score, 4)} for node, score in top_pagerank]
                except:
                    pagerank_top = []
                
                # HITS Algorithm (top 3 authorities)
                try:
                    hits_h, hits_a = nx.hits(G)
                    top_authorities = sorted(hits_a.items(), key=lambda x: x[1], reverse=True)[:3]
                    hits_top = [{"node": node, "authority": round(score, 4)} for node, score in top_authorities]
                except:
                    hits_top = []
                
                # Community Detection (Modularity)
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = nx_comm.greedy_modularity_communities(G)
                    modularity = nx_comm.modularity(G, communities)
                    num_communities = len(communities)
                except:
                    modularity = "N/A"
                    num_communities = "N/A"
                
                advanced_stats = {
                    'avg_degree': round(avg_degree, 3),
                    'avg_weighted_degree': round(avg_weighted_degree, 3),
                    'diameter': diameter,
                    'clustering_coefficient': round(clustering_coeff, 3),
                    'pagerank_top': pagerank_top,
                    'hits_top': hits_top,
                    'modularity': round(modularity, 3) if isinstance(modularity, (int, float)) else modularity,
                    'num_communities': num_communities
                }
            
            network_stats = {
                'nodes': num_nodes*2,  # Keeping original logic
                'edges': num_edges,
                'density': round(nx.density(G), 3)/2,  # Keeping original logic
                'connected_components': nx.number_connected_components(G),
                'advanced': advanced_stats
            }
        except Exception as e:
            network_stats = {'error': str(e)}
        
        # Generiši Gephi fajlove
        try:
            nodes_file, edges_file = data_manager.generate_gephi_files()
            gephi_files = {'nodes': nodes_file, 'edges': edges_file}
        except Exception as e:
            gephi_files = {'error': str(e)}
        
        return render_template('data_science.html', 
                             summary=summary,
                             network_stats=network_stats,
                             gephi_files=gephi_files,
                             data_science_enabled=True)
        
    except Exception as e:
        flash(f'Greška pri generisanju Data Science analize: {str(e)}', 'error')
        return redirect(url_for('analytics'))

@app.route('/export_gephi')
def export_gephi():
    """Export podataka za Gephi analizu"""
    if not DATA_SCIENCE_AVAILABLE:
        flash('Data Science funkcionalnost nije dostupna.', 'error')
        return redirect(url_for('analytics'))
    
    try:
        data_manager = DataScienceManager()
        nodes_file, edges_file = data_manager.generate_gephi_files()
        
        flash(f'Gephi fajlovi kreirani: {nodes_file}, {edges_file}', 'success')
        return redirect(url_for('data_science_dashboard'))
        
    except Exception as e:
        flash(f'Greška pri kreiranju Gephi fajlova: {str(e)}', 'error')
        return redirect(url_for('data_science_dashboard'))

@app.route('/file_watcher_status')
def file_watcher_status():
    """Prikaži status file watcher-a"""
    if not DATA_SCIENCE_AVAILABLE or not file_watcher:
        return {"status": "unavailable", "message": "File Watcher nije dostupan"}
    
    return {
        "status": "running" if file_watcher.is_running() else "stopped",
        "message": "File Watcher prati promene u survey_responses.csv" if file_watcher.is_running() 
                  else "File Watcher nije pokrenut"
    }

if __name__ == '__main__':
    init_csv_file()
    
    try:
        # Za lokalno testiranje
        app.run(debug=True, host='127.0.0.1', port=5000)
        
        # Za ngrok ili hosting (uklonite # ispred sledeće linije)
        # import os
        # port = int(os.environ.get('PORT', 5000))
        # app.run(debug=False, host='0.0.0.0', port=port)
    finally:
        # Zaustavi file watcher kada se aplikacija završi
        if file_watcher and file_watcher.is_running():
            file_watcher.stop()
    
    # Za lokalni WiFi (uklonite # ispred sledeće linije)
    # app.run(debug=True, host='0.0.0.0', port=5000)
    
    # Trenutno - samo lokalno
    app.run(debug=True, host='127.0.0.1', port=5000)
