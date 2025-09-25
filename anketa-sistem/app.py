from flask import Flask, render_template, request, redirect, url_for, flash, session
import csv
import os
from datetime import datetime
import re
import difflib
from collections import Counter
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# File path for storing survey responses
RESPONSES_FILE = 'survey_responses.csv'

# Funkcije za poređenje teksta
def clean_text(text):
    """Očisti tekst od nepotrebnih karaktera i normalizuj"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # Ukloni interpunkciju
    text = re.sub(r'\s+', ' ', text)      # Normalizuj razmake
    return text.strip()

def calculate_similarity_metrics(text1, text2):
    """Izračunaj različite metrike sličnosti između dva teksta"""
    
    # Očisti tekstove
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    # 1. Jaccard similarity (na nivou reči)
    words1 = set(clean1.split())
    words2 = set(clean2.split())
    jaccard = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
    
    # 2. Cosine similarity
    def cosine_similarity(text1, text2):
        # Kreiraj vektore reči
        words = list(set(text1.split() + text2.split()))
        vec1 = [text1.split().count(word) for word in words]
        vec2 = [text2.split().count(word) for word in words]
        
        # Izračunaj cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)
    
    cosine = cosine_similarity(clean1, clean2)
    
    # 3. Sequence Matcher (SequenceMatcher ratio)
    sequence_ratio = difflib.SequenceMatcher(None, clean1, clean2).ratio()
    
    # 4. Word overlap percentage
    total_words1 = len(clean1.split())
    total_words2 = len(clean2.split())
    common_words = len(words1.intersection(words2))
    word_overlap = (2 * common_words) / (total_words1 + total_words2) if (total_words1 + total_words2) > 0 else 0
    
    # 5. Length ratio
    length_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0
    
    # 6. Line-by-line comparison
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    line_similarity = difflib.SequenceMatcher(None, lines1, lines2).ratio()
    
    return {
        'jaccard_similarity': round(jaccard * 100, 2),
        'cosine_similarity': round(cosine * 100, 2),
        'sequence_similarity': round(sequence_ratio * 100, 2),
        'word_overlap': round(word_overlap * 100, 2),
        'length_ratio': round(length_ratio * 100, 2),
        'line_similarity': round(line_similarity * 100, 2),
        'average_similarity': round((jaccard + cosine + sequence_ratio + word_overlap + length_ratio + line_similarity) / 6 * 100, 2)
    }

def get_detailed_diff(text1, text2):
    """Generiši detaljnu analizu razlika između tekstova"""
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    diff = list(difflib.unified_diff(lines1, lines2, 
                                   fromfile='Originalni README', 
                                   tofile='AI Generisani README',
                                   lineterm=''))
    
    return '\n'.join(diff)

def interpret_similarity_results(metrics):
    """Generiši zaključke na osnovu metrika sličnosti"""
    avg_sim = metrics['average_similarity']
    interpretations = []
    
    # Ukupna ocena
    if avg_sim >= 90:
        interpretations.append({
            'category': 'Ukupna Ocena',
            'level': 'very_high',
            'title': '🔥 Izuzetno Visoka Sličnost (90%+)',
            'description': 'AI je skoro identično reprodukovao originalni sadržaj. Ovo može ukazati na vrlo precizno razumevanje ili možda previše doslovno kopiranje.',
            'implications': ['Vrlo pouzdano generisanje', 'Možda nedostaje kreativnost', 'Terminologija potpuno konzistentna']
        })
    elif avg_sim >= 75:
        interpretations.append({
            'category': 'Ukupna Ocena',
            'level': 'high',
            'title': '✅ Visoka Sličnost (75-89%)',
            'description': 'AI je uspešno zadržao ključne elemente originalnog teksta uz neke modifikacije. Odličan balans između konzistentnosti i adaptacije.',
            'implications': ['Dobra preciznost', 'Zadržana struktura', 'Umerenost u izmjenama']
        })
    elif avg_sim >= 50:
        interpretations.append({
            'category': 'Ukupna Ocena', 
            'level': 'medium',
            'title': '⚖️ Umerena Sličnost (50-74%)',
            'description': 'AI je zadržao osnovnu temu ali je napravio značajne izmene u pristupu, stilu ili strukturi.',
            'implications': ['Umeren nivo adaptacije', 'Možda poboljšanja u organizaciji', 'Potrebna verifikacija sadržaja']
        })
    else:
        interpretations.append({
            'category': 'Ukupna Ocena',
            'level': 'low', 
            'title': '⚠️ Niska Sličnost (<50%)',
            'description': 'AI je značajno promenio pristup ili možda nije u potpunosti razumeo originalni sadržaj.',
            'implications': ['Potrebna pažljiva revizija', 'Možda kreativniji pristup', 'Verifikacija tačnosti informacija']
        })
    
    # Analiza specifičnih metrika
    if metrics['jaccard_similarity'] >= 80:
        interpretations.append({
            'category': 'Terminologija',
            'level': 'high',
            'title': '📝 Konzistentna Terminologija',
            'description': f"Jaccard sličnost od {metrics['jaccard_similarity']}% ukazuje na to da AI koristi iste ključne reči i termine.",
            'implications': ['Terminologija je pouzdana', 'Ključni pojmovi su zadržani', 'Dobra konzistentnost vokabulara']
        })
    
    if metrics['cosine_similarity'] >= 85:
        interpretations.append({
            'category': 'Tematska Sličnost',
            'level': 'high', 
            'title': '🎯 Odlično Razumevanje Teme',
            'description': f"Cosine sličnost od {metrics['cosine_similarity']}% pokazuje da je AI razumeo centralnu temu i fokus dokumenta.",
            'implications': ['AI razume kontekst', 'Tematski fokus je zadržan', 'Dobra distribucija važnosti pojmova']
        })
    
    if metrics['line_similarity'] >= 70:
        interpretations.append({
            'category': 'Struktura',
            'level': 'high',
            'title': '🏗️ Zadržana Struktura',
            'description': f"Sličnost linija od {metrics['line_similarity']}% ukazuje na to da je AI poštovao organizaciju originalnog dokumenta.",
            'implications': ['Logična organizacija zadržana', 'Formatiranje poštovano', 'Dobra struktura sadržaja']
        })
    
    if metrics['word_overlap'] >= 75:
        interpretations.append({
            'category': 'Preklapanje Sadržaja',
            'level': 'high',
            'title': '🔄 Visoko Preklapanje Reči',
            'description': f"Preklapanje reči od {metrics['word_overlap']}% može ukazati na vrlo doslovno preuzimanje sadržaja.",
            'implications': ['Visoka preciznost', 'Možda previše doslovno', 'Konzistentnost naziva']
        })
    
    # Upozorenja
    if avg_sim >= 95:
        interpretations.append({
            'category': 'Upozorenje',
            'level': 'warning',
            'title': '🚨 Previše Visoka Sličnost',
            'description': 'Sličnost preko 95% može ukazati na skoro identično kopiranje. Proverite da li je AI dodao vrednost.',
            'implications': ['Možda plagijat', 'Nedostatak kreativnosti', 'Proverite originalnost']
        })
    
    return interpretations

def save_ai_comparison_data(data):
    """Sačuvaj podatke o poređenju AI alata u CSV fajl"""
    comparison_file = 'ai_comparison_responses.csv'
    
    # Kreiraj CSV fajl sa headerima ako ne postoji
    if not os.path.exists(comparison_file):
        with open(comparison_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'task_type', 'task_description',
                'chatgpt_solution', 'chatgpt_rating_quality', 'chatgpt_rating_accuracy', 'chatgpt_rating_usefulness', 'chatgpt_comments',
                'copilot_solution', 'copilot_rating_quality', 'copilot_rating_accuracy', 'copilot_rating_usefulness', 'copilot_comments',
                'overall_preference', 'preference_reason'
            ])
    
    # Dodaj novi red
    with open(comparison_file, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            data['timestamp'], data['task_type'], data['task_description'],
            data['chatgpt_solution'], data['chatgpt_rating_quality'], data['chatgpt_rating_accuracy'], data['chatgpt_rating_usefulness'], data['chatgpt_comments'],
            data['copilot_solution'], data['copilot_rating_quality'], data['copilot_rating_accuracy'], data['copilot_rating_usefulness'], data['copilot_comments'],
            data['overall_preference'], data['preference_reason']
        ])

def analyze_ai_comparison(data, similarity_metrics):
    """Analiziraj rezultate poređenja AI alata"""
    analysis = {}
    
    # Izračunaj prosečne ocene
    try:
        chatgpt_avg = (
            int(data['chatgpt_rating_quality'] or 0) + 
            int(data['chatgpt_rating_accuracy'] or 0) + 
            int(data['chatgpt_rating_usefulness'] or 0)
        ) / 3 if data['chatgpt_rating_quality'] else 0
        
        copilot_avg = (
            int(data['copilot_rating_quality'] or 0) + 
            int(data['copilot_rating_accuracy'] or 0) + 
            int(data['copilot_rating_usefulness'] or 0)
        ) / 3 if data['copilot_rating_quality'] else 0
    except:
        chatgpt_avg = 0
        copilot_avg = 0
    
    analysis['chatgpt_average'] = round(chatgpt_avg, 2)
    analysis['copilot_average'] = round(copilot_avg, 2)
    analysis['score_difference'] = round(abs(chatgpt_avg - copilot_avg), 2)
    
    # Određuj pobednika na osnovu ocena
    if chatgpt_avg > copilot_avg:
        analysis['score_winner'] = 'ChatGPT'
        analysis['score_advantage'] = round(chatgpt_avg - copilot_avg, 2)
    elif copilot_avg > chatgpt_avg:
        analysis['score_winner'] = 'Copilot'
        analysis['score_advantage'] = round(copilot_avg - chatgpt_avg, 2)
    else:
        analysis['score_winner'] = 'Nerešeno'
        analysis['score_advantage'] = 0
    
    # Analiza sličnosti
    if similarity_metrics['average_similarity'] >= 80:
        analysis['similarity_level'] = 'Vrlo slična rešenja'
        analysis['similarity_interpretation'] = 'Oba AI alata su dala vrlo slična rešenja, razlike su minimalne.'
    elif similarity_metrics['average_similarity'] >= 60:
        analysis['similarity_level'] = 'Umereno slična rešenja'
        analysis['similarity_interpretation'] = 'Rešenja dele zajedničke elemente ali imaju različite pristupe.'
    else:
        analysis['similarity_level'] = 'Različita rešenja'
        analysis['similarity_interpretation'] = 'AI alati su dali značajno različite pristupe istom problemu.'
    
    # Analiza po kategorijama
    categories = ['quality', 'accuracy', 'usefulness']
    for category in categories:
        chatgpt_score = int(data.get(f'chatgpt_rating_{category}', 0) or 0)
        copilot_score = int(data.get(f'copilot_rating_{category}', 0) or 0)
        
        if chatgpt_score > copilot_score:
            analysis[f'{category}_winner'] = 'ChatGPT'
        elif copilot_score > chatgpt_score:
            analysis[f'{category}_winner'] = 'Copilot'
        else:
            analysis[f'{category}_winner'] = 'Nerešeno'
    
    return analysis

def load_survey_data():
    """Učitaj podatke iz ankete kao pandas DataFrame"""
    if not os.path.exists(RESPONSES_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(RESPONSES_FILE, encoding='utf-8')
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Greška pri učitavanju podataka: {e}")
        return pd.DataFrame()

def load_ai_comparison_data():
    """Učitaj podatke iz AI poređenja kao pandas DataFrame"""
    comparison_file = 'ai_comparison_responses.csv'
    if not os.path.exists(comparison_file):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(comparison_file, encoding='utf-8')
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Konvertuj ocene u numeričke vrednosti
            rating_cols = ['chatgpt_rating_quality', 'chatgpt_rating_accuracy', 'chatgpt_rating_usefulness',
                          'copilot_rating_quality', 'copilot_rating_accuracy', 'copilot_rating_usefulness']
            for col in rating_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Greška pri učitavanju AI comparison podataka: {e}")
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
    if 'radni_odnos' in survey_df.columns:
        employed_count = (survey_df['radni_odnos'] == 'da').sum()
    employed_percentage = (employed_count / total_responses * 100) if total_responses > 0 else 0
    
    analytics['summary_stats'] = {
        'total_responses': total_responses,
        'total_survey_responses': total_responses,  # Template očekuje ovaj ključ
        'total_ai_comparisons': 0,  # Za kompatibilnost sa template-om
        'avg_age': 2025 - survey_df['godina_rodjenja'].astype(int).mean() if 'godina_rodjenja' in survey_df.columns else 0,
        'countries': survey_df['drzava'].nunique() if 'drzava' in survey_df.columns else 0,
        'completion_rate': 100.0,  # Svi odgovori su kompletni
        'employed_percentage': employed_percentage,
        'chatgpt_aware': 0,  # Ove metrike ću dodati later ako postoje u podacima
        'copilot_aware': 0,
        'avg_programming_env': 0,  # Uklanjam pokušaj konverzije string polja
        'avg_programming_lang': 0,  # Uklanjam pokušaj konverzije string polja
        'avg_ai_usage': survey_df['generativni_ai_poznavanje'].astype(float).mean() if 'generativni_ai_poznavanje' in survey_df.columns and survey_df['generativni_ai_poznavanje'].notna().any() else 0
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
    
    # 2. AI POZNAVANJE I KORIŠĆENJE
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Generativni AI poznavanje
    if 'generativni_ai_poznavanje' in survey_df.columns:
        ai_knowledge = survey_df['generativni_ai_poznavanje'].value_counts().sort_index()
        ai_knowledge.plot(kind='bar', ax=ax1, color='#8e44ad', alpha=0.8)
        ax1.set_title('Poznavanje Generativnih AI Alata', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Broj učesnika')
        ax1.set_xlabel('Ocena (1=loše, 5=odlično)')
    
    # Prompt engineering
    if 'prompt_engineering' in survey_df.columns:
        prompt_skills = survey_df['prompt_engineering'].value_counts().sort_index()
        prompt_skills.plot(kind='bar', ax=ax2, color='#27ae60', alpha=0.8)
        ax2.set_title('Veštine Prompt Engineering-a', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Broj učesnika')
        ax2.set_xlabel('Ocena (1=loše, 5=odlično)')
    
    # Poznati AI alati - analiza najčešćih
    if 'poznati_ai_alati' in survey_df.columns:
        all_tools = []
        for tools in survey_df['poznati_ai_alati'].dropna():
            if tools:
                all_tools.extend([tool.strip() for tool in tools.split(',')])
        
        if all_tools:
            tool_counts = pd.Series(all_tools).value_counts().head(8)
            tool_counts.plot(kind='barh', ax=ax3, color='#f39c12', alpha=0.8)
            ax3.set_title('Najpoznatiji AI Alati', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Broj spomenuta')
    
    # Svrhe korišćenja
    if 'svrhe_koriscenja' in survey_df.columns:
        all_purposes = []
        for purposes in survey_df['svrhe_koriscenja'].dropna():
            if purposes:
                all_purposes.extend([purpose.strip() for purpose in purposes.split(',')])
        
        if all_purposes:
            purpose_counts = pd.Series(all_purposes).value_counts().head(6)
            purpose_counts.plot(kind='bar', ax=ax4, color='#e67e22', alpha=0.8)
            ax4.set_title('Svrhe Korišćenja AI Alata', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Broj spomenuta')
            ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    analytics['survey_charts'].append({
        'title': 'AI Znanje i Korišćenje',
        'image': create_matplotlib_chart(fig)
    })
    
    # 3. KVIZ ANALIZA
    quiz_fields = [
        'chatgpt_omni', 'copilot_task', 'copilot_chat', 'google_model',
        'gpt_realtime', 'codex_successor', 'chatgpt_data_analysis', 'copilot_workspace',
        'anthropic_model', 'creativity_parameter', 'transformer_basis', 'university_guidelines'
    ]
    
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
            'grana_oblast': request.form.get('grana_oblast', ''),
            'godine_staza': request.form.get('godine_staza', ''),
            'institucija': request.form.get('institucija', ''),
            'uza_oblast': request.form.get('uza_oblast', ''),
            'pisanje_softvera': request.form.get('pisanje_softvera', ''),
            'generativni_ai_poznavanje': request.form.get('generativni_ai_poznavanje', ''),
            'programski_jezici': request.form.get('programski_jezici', ''),
            'programska_okruzenja': request.form.get('programska_okruzenja', ''),
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
                data['grana_oblast'],
                data['godine_staza'],
                data['institucija'],
                data['uza_oblast'],
                data['pisanje_softvera'],
                data['generativni_ai_poznavanje'],
                data['programski_jezici'],
                data['programska_okruzenja'],
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

@app.route('/compare')
def compare_docs():
    """Stranica za poređenje README dokumentacije"""
    return render_template('compare.html')

@app.route('/ai_comparison')
def ai_comparison():
    """Stranica za uporedno testiranje ChatGPT vs Copilot"""
    return render_template('ai_comparison.html')

@app.route('/ai_comparison_submit', methods=['POST'])
def ai_comparison_submit():
    """Obradi uporedno testiranje ChatGPT vs Copilot"""
    try:
        # Osnovne informacije
        task_type = request.form.get('task_type', '')
        task_description = request.form.get('task_description', '').strip()
        
        # ChatGPT odgovor i ocena
        chatgpt_solution = request.form.get('chatgpt_solution', '').strip()
        chatgpt_rating_quality = request.form.get('chatgpt_rating_quality', '')
        chatgpt_rating_accuracy = request.form.get('chatgpt_rating_accuracy', '')
        chatgpt_rating_usefulness = request.form.get('chatgpt_rating_usefulness', '')
        chatgpt_comments = request.form.get('chatgpt_comments', '').strip()
        
        # Copilot odgovor i ocena
        copilot_solution = request.form.get('copilot_solution', '').strip()
        copilot_rating_quality = request.form.get('copilot_rating_quality', '')
        copilot_rating_accuracy = request.form.get('copilot_rating_accuracy', '')
        copilot_rating_usefulness = request.form.get('copilot_rating_usefulness', '')
        copilot_comments = request.form.get('copilot_comments', '').strip()
        
        # Opšte poređenje
        overall_preference = request.form.get('overall_preference', '')
        preference_reason = request.form.get('preference_reason', '').strip()
        
        if not all([task_description, chatgpt_solution, copilot_solution]):
            flash('Molimo unesite opis zadatka i oba rešenja.', 'error')
            return redirect(url_for('ai_comparison'))
        
        # Sačuvaj podatke u CSV
        comparison_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task_type': task_type,
            'task_description': task_description,
            'chatgpt_solution': chatgpt_solution,
            'chatgpt_rating_quality': chatgpt_rating_quality,
            'chatgpt_rating_accuracy': chatgpt_rating_accuracy,
            'chatgpt_rating_usefulness': chatgpt_rating_usefulness,
            'chatgpt_comments': chatgpt_comments,
            'copilot_solution': copilot_solution,
            'copilot_rating_quality': copilot_rating_quality,
            'copilot_rating_accuracy': copilot_rating_accuracy,
            'copilot_rating_usefulness': copilot_rating_usefulness,
            'copilot_comments': copilot_comments,
            'overall_preference': overall_preference,
            'preference_reason': preference_reason
        }
        
        # Sačuvaj u poseban CSV fajl
        save_ai_comparison_data(comparison_data)
        
        # Izračunaj metrike sličnosti između rešenja
        similarity_metrics = calculate_similarity_metrics(chatgpt_solution, copilot_solution)
        
        # Analiziraj ocene
        comparison_analysis = analyze_ai_comparison(comparison_data, similarity_metrics)
        
        return render_template('ai_comparison_results.html',
                             data=comparison_data,
                             metrics=similarity_metrics,
                             analysis=comparison_analysis)
        
    except Exception as e:
        flash(f'Greška pri obradi poređenja: {str(e)}', 'error')
        return redirect(url_for('ai_comparison'))

@app.route('/ai_comparison_results')
def ai_comparison_results():
    """Prikaži sve rezultate AI poređenja"""
    try:
        comparison_file = 'ai_comparison_responses.csv'
        comparisons = []
        
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                comparisons = list(reader)
        
        # Osnovne statistike
        total_comparisons = len(comparisons)
        chatgpt_wins = sum(1 for c in comparisons if c.get('overall_preference') == 'chatgpt')
        copilot_wins = sum(1 for c in comparisons if c.get('overall_preference') == 'copilot')
        ties = sum(1 for c in comparisons if c.get('overall_preference') == 'equal')
        
        # Prosečne ocene
        chatgpt_ratings = []
        copilot_ratings = []
        
        for comp in comparisons:
            try:
                if comp.get('chatgpt_rating_quality'):
                    chatgpt_avg = (
                        int(comp['chatgpt_rating_quality']) + 
                        int(comp['chatgpt_rating_accuracy']) + 
                        int(comp['chatgpt_rating_usefulness'])
                    ) / 3
                    chatgpt_ratings.append(chatgpt_avg)
                
                if comp.get('copilot_rating_quality'):
                    copilot_avg = (
                        int(comp['copilot_rating_quality']) + 
                        int(comp['copilot_rating_accuracy']) + 
                        int(comp['copilot_rating_usefulness'])
                    ) / 3
                    copilot_ratings.append(copilot_avg)
            except:
                continue
        
        stats = {
            'total_comparisons': total_comparisons,
            'chatgpt_wins': chatgpt_wins,
            'copilot_wins': copilot_wins,
            'ties': ties,
            'chatgpt_win_rate': round((chatgpt_wins / total_comparisons * 100) if total_comparisons > 0 else 0, 1),
            'copilot_win_rate': round((copilot_wins / total_comparisons * 100) if total_comparisons > 0 else 0, 1),
            'chatgpt_avg_rating': round(sum(chatgpt_ratings) / len(chatgpt_ratings) if chatgpt_ratings else 0, 2),
            'copilot_avg_rating': round(sum(copilot_ratings) / len(copilot_ratings) if copilot_ratings else 0, 2)
        }
        
        return render_template('ai_comparison_all_results.html', comparisons=comparisons, stats=stats)
        
    except Exception as e:
        flash(f'Greška pri učitavanju rezultata: {str(e)}', 'error')
        return redirect(url_for('ai_comparison'))

@app.route('/analytics')
def analytics():
    """Napredna analitika i vizualizacije podataka"""
    try:
        analytics_data = generate_analytics_data()
        return render_template('analytics.html', analytics=analytics_data)
    except Exception as e:
        flash(f'Greška pri generisanju analitike: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/compare_submit', methods=['POST'])
def compare_submit():
    """Obradi poređenje README fajlova"""
    try:
        original_readme = request.form.get('original_readme', '').strip()
        ai_readme = request.form.get('ai_readme', '').strip()
        
        if not original_readme or not ai_readme:
            flash('Molimo unesite oba teksta za poređenje.', 'error')
            return redirect(url_for('compare_docs'))
        
        # Izračunaj metrike sličnosti
        similarity_metrics = calculate_similarity_metrics(original_readme, ai_readme)
        
        # Generiši interpretacije rezultata
        interpretations = interpret_similarity_results(similarity_metrics)
        
        # Generiši detaljnu analizu
        detailed_diff = get_detailed_diff(original_readme, ai_readme)
        
        # Dodaj osnovne statistike
        stats = {
            'original_length': len(original_readme),
            'ai_length': len(ai_readme),
            'original_words': len(original_readme.split()),
            'ai_words': len(ai_readme.split()),
            'original_lines': len(original_readme.split('\n')),
            'ai_lines': len(ai_readme.split('\n'))
        }
        
        return render_template('compare_results.html', 
                             metrics=similarity_metrics,
                             interpretations=interpretations,
                             diff=detailed_diff,
                             stats=stats,
                             original=original_readme,
                             ai_generated=ai_readme)
        
    except Exception as e:
        flash(f'Greška pri poređenju: {str(e)}', 'error')
        return redirect(url_for('compare_docs'))

if __name__ == '__main__':
    init_csv_file()
    # Za lokalno testiranje
    # app.run(debug=True, host='127.0.0.1', port=5000)
    
    # Za ngrok ili hosting (uklonite # ispred sledeće linije)
    # import os
    # port = int(os.environ.get('PORT', 5000))
    # app.run(debug=False, host='0.0.0.0', port=port)
    
    # Za lokalni WiFi (uklonite # ispred sledeće linije)
    # app.run(debug=True, host='0.0.0.0', port=5000)
    
    # Trenutno - samo lokalno
    app.run(debug=True, host='127.0.0.1', port=5000)
