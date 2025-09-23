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
    ai_comparison_df = load_ai_comparison_data()
    
    analytics = {
        'survey_charts': [],
        'ai_comparison_charts': [],
        'plotly_charts': [],
        'summary_stats': {}
    }
    
    # === ANKETA ANALYTICS ===
    if not survey_df.empty:
        # 1. AI korišćenje po godinama studija
        plt.style.use('seaborn-v0_8')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ChatGPT korišćenje
        chatgpt_by_year = survey_df.groupby('godina_studija')['koristi_chatgpt'].apply(
            lambda x: (x == 'da').sum() / len(x) * 100
        )
        chatgpt_by_year.plot(kind='bar', ax=ax1, color='#10a37f', alpha=0.8)
        ax1.set_title('ChatGPT Korišćenje po Godinama Studija', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Procenat korisnika (%)')
        ax1.set_xlabel('Godina studija')
        ax1.tick_params(axis='x', rotation=45)
        
        # Copilot korišćenje
        copilot_by_year = survey_df.groupby('godina_studija')['koristi_copilot'].apply(
            lambda x: (x == 'da').sum() / len(x) * 100
        )
        copilot_by_year.plot(kind='bar', ax=ax2, color='#0969da', alpha=0.8)
        ax2.set_title('Copilot Korišćenje po Godinama Studija', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Procenat korisnika (%)')
        ax2.set_xlabel('Godina studija')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        analytics['survey_charts'].append({
            'title': 'AI Korišćenje po Godinama Studija',
            'image': create_matplotlib_chart(fig)
        })
        
        # 2. Učestalost korišćenja - pie chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ChatGPT učestalost
        chatgpt_freq = survey_df['chatgpt_cesto'].value_counts()
        colors1 = ['#10a37f', '#20c997', '#6f42c1', '#fd7e14', '#dc3545']
        ax1.pie(chatgpt_freq.values, labels=chatgpt_freq.index, autopct='%1.1f%%', 
                colors=colors1, startangle=90)
        ax1.set_title('ChatGPT - Učestalost Korišćenja', fontsize=14, fontweight='bold')
        
        # Copilot učestalost
        copilot_freq = survey_df['copilot_cesto'].value_counts()
        colors2 = ['#0969da', '#20c997', '#6f42c1', '#fd7e14', '#dc3545']
        ax2.pie(copilot_freq.values, labels=copilot_freq.index, autopct='%1.1f%%',
                colors=colors2, startangle=90)
        ax2.set_title('Copilot - Učestalost Korišćenja', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        analytics['survey_charts'].append({
            'title': 'Učestalost Korišćenja AI Alata',
            'image': create_matplotlib_chart(fig)
        })
        
        # 3. Preporuke drugim studentima
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        preporuke = survey_df['preporuka_drugim'].value_counts()
        colors = ['#28a745', '#dc3545', '#ffc107']
        bars = ax.bar(preporuke.index, preporuke.values, color=colors, alpha=0.8)
        ax.set_title('Da li biste preporučili AI alate drugim studentima?', fontsize=14, fontweight='bold')
        ax.set_ylabel('Broj odgovora')
        ax.set_xlabel('Odgovor')
        
        # Dodaj tekstove na stubove
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        analytics['survey_charts'].append({
            'title': 'Preporuke AI Alata Drugim Studentima',
            'image': create_matplotlib_chart(fig)
        })
    
    # === AI COMPARISON ANALYTICS ===
    if not ai_comparison_df.empty:
        # 4. Poređenje prosečnih ocena
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Izračunaj prosečne ocene po kategorijama
        categories = ['quality', 'accuracy', 'usefulness']
        chatgpt_scores = []
        copilot_scores = []
        
        for cat in categories:
            chatgpt_col = f'chatgpt_rating_{cat}'
            copilot_col = f'copilot_rating_{cat}'
            if chatgpt_col in ai_comparison_df.columns and copilot_col in ai_comparison_df.columns:
                chatgpt_scores.append(ai_comparison_df[chatgpt_col].mean())
                copilot_scores.append(ai_comparison_df[copilot_col].mean())
        
        x = range(len(categories))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], chatgpt_scores, width, label='ChatGPT', 
               color='#10a37f', alpha=0.8)
        ax.bar([i + width/2 for i in x], copilot_scores, width, label='Copilot', 
               color='#0969da', alpha=0.8)
        
        ax.set_title('Prosečne Ocene: ChatGPT vs Copilot', fontsize=14, fontweight='bold')
        ax.set_ylabel('Prosečna ocena (1-5)')
        ax.set_xlabel('Kategorija')
        ax.set_xticks(x)
        ax.set_xticklabels(['Kvalitet', 'Tačnost', 'Korisnost'])
        ax.legend()
        ax.set_ylim(0, 5)
        
        # Dodaj vrednosti na stubove
        for i, (c_score, co_score) in enumerate(zip(chatgpt_scores, copilot_scores)):
            ax.text(i - width/2, c_score + 0.05, f'{c_score:.2f}', ha='center', fontweight='bold')
            ax.text(i + width/2, co_score + 0.05, f'{co_score:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        analytics['ai_comparison_charts'].append({
            'title': 'ChatGPT vs Copilot - Prosečne Ocene',
            'image': create_matplotlib_chart(fig)
        })
        
        # 5. Preferencije korisnika
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        preferences = ai_comparison_df['overall_preference'].value_counts()
        colors = ['#10a37f', '#0969da', '#ffc107']
        wedges, texts, autotexts = ax.pie(preferences.values, labels=preferences.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Ukupne Preferencije: ChatGPT vs Copilot', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        analytics['ai_comparison_charts'].append({
            'title': 'Korisničke Preferencije AI Alata',
            'image': create_matplotlib_chart(fig)
        })
    
    # === PLOTLY INTERAKTIVNI GRAFIKONI ===
    if not survey_df.empty:
        # Timeline analiza
        daily_responses = survey_df.groupby(survey_df['timestamp'].dt.date).size().reset_index()
        daily_responses.columns = ['date', 'responses']
        
        fig = px.line(daily_responses, x='date', y='responses', 
                     title='Broj Odgovora po Danima',
                     labels={'date': 'Datum', 'responses': 'Broj Odgovora'})
        fig.update_layout(title_font_size=16)
        
        analytics['plotly_charts'].append({
            'title': 'Timeline Odgovora',
            'chart': json.dumps(fig, cls=PlotlyJSONEncoder)
        })
    
    # === SUMMARY STATISTIKE ===
    analytics['summary_stats'] = {
        'total_survey_responses': len(survey_df) if not survey_df.empty else 0,
        'total_ai_comparisons': len(ai_comparison_df) if not ai_comparison_df.empty else 0,
        'chatgpt_users': (survey_df['koristi_chatgpt'] == 'da').sum() if not survey_df.empty else 0,
        'copilot_users': (survey_df['koristi_copilot'] == 'da').sum() if not survey_df.empty else 0,
        'recommendation_rate': (survey_df['preporuka_drugim'] == 'da').sum() / len(survey_df) * 100 if not survey_df.empty else 0
    }
    
    return analytics

def init_csv_file():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(RESPONSES_FILE):
        with open(RESPONSES_FILE, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp',
                'godina_studija',
                'smer',
                'koristi_chatgpt',
                'chatgpt_cesto',
                'chatgpt_svrha',
                'koristi_copilot',
                'copilot_cesto',
                'copilot_svrha',
                'uticaj_na_ucenje',
                'prednosti',
                'nedostaci',
                'preporuka_drugim'
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
            'godina_studija': request.form.get('godina_studija', ''),
            'smer': request.form.get('smer', ''),
            'koristi_chatgpt': request.form.get('koristi_chatgpt', ''),
            'chatgpt_cesto': request.form.get('chatgpt_cesto', ''),
            'chatgpt_svrha': request.form.get('chatgpt_svrha', ''),
            'koristi_copilot': request.form.get('koristi_copilot', ''),
            'copilot_cesto': request.form.get('copilot_cesto', ''),
            'copilot_svrha': request.form.get('copilot_svrha', ''),
            'uticaj_na_ucenje': request.form.get('uticaj_na_ucenje', ''),
            'prednosti': request.form.get('prednosti', ''),
            'nedostaci': request.form.get('nedostaci', ''),
            'preporuka_drugim': request.form.get('preporuka_drugim', '')
        }
        
        # Save to CSV
        with open(RESPONSES_FILE, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                data['timestamp'],
                data['godina_studija'],
                data['smer'],
                data['koristi_chatgpt'],
                data['chatgpt_cesto'],
                data['chatgpt_svrha'],
                data['koristi_copilot'],
                data['copilot_cesto'],
                data['copilot_svrha'],
                data['uticaj_na_ucenje'],
                data['prednosti'],
                data['nedostaci'],
                data['preporuka_drugim']
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
        
        # Basic statistics
        total_responses = len(responses)
        chatgpt_users = sum(1 for r in responses if r.get('koristi_chatgpt') == 'da')
        copilot_users = sum(1 for r in responses if r.get('koristi_copilot') == 'da')
        
        stats = {
            'total_responses': total_responses,
            'chatgpt_users': chatgpt_users,
            'copilot_users': copilot_users,
            'chatgpt_percentage': round((chatgpt_users / total_responses * 100) if total_responses > 0 else 0, 1),
            'copilot_percentage': round((copilot_users / total_responses * 100) if total_responses > 0 else 0, 1)
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
