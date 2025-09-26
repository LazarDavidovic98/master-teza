from flask import Flask, render_template, request, redirect, url_for, flash, session
import csv
import os
from datetime import datetime
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
    
    # Bezbedna konverzija AI znanja u numeričke vrednosti
    avg_ai_usage = 0
    if 'generativni_ai_poznavanje' in survey_df.columns:
        # Konvertuj u numeričke vrednosti, ignoriši nevalidne
        numeric_ai_knowledge = pd.to_numeric(survey_df['generativni_ai_poznavanje'], errors='coerce')
        valid_scores = numeric_ai_knowledge.dropna()
        if not valid_scores.empty:
            avg_ai_usage = round(valid_scores.mean(), 2)
    
    analytics['summary_stats'] = {
        'total_responses': total_responses,
        'total_survey_responses': total_responses,  # Template očekuje ovaj ključ
        'avg_age': round(2025 - pd.to_numeric(survey_df['godina_rodjenja'], errors='coerce').mean(), 1) if 'godina_rodjenja' in survey_df.columns else 0,
        'countries': survey_df['drzava'].nunique() if 'drzava' in survey_df.columns else 0,
        'completion_rate': 100.0,  # Svi odgovori su kompletni
        'employed_percentage': round(employed_percentage, 1),
        'chatgpt_aware': 0,  # Ove metrike ću dodati later ako postoje u podacima
        'copilot_aware': 0,
        'avg_programming_env': 0,  # Uklanjam pokušaj konverzije string polja
        'avg_programming_lang': 0,  # Uklanjam pokušaj konverzije string polja
        'avg_ai_usage': avg_ai_usage
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
    
    # Generativni AI poznavanje - samo numeričke vrednosti
    if 'generativni_ai_poznavanje' in survey_df.columns:
        # Konvertuj u numeričke vrednosti i filtiraj nevalidne
        numeric_ai_knowledge = pd.to_numeric(survey_df['generativni_ai_poznavanje'], errors='coerce')
        valid_ai_knowledge = numeric_ai_knowledge.dropna()
        if not valid_ai_knowledge.empty:
            ai_knowledge = valid_ai_knowledge.value_counts().sort_index()
            ai_knowledge.plot(kind='bar', ax=ax1, color='#8e44ad', alpha=0.8)
            ax1.set_title('Poznavanje Generativnih AI Alata', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Broj učesnika')
            ax1.set_xlabel('Ocena (1=loše, 5=odlično)')
    
    # Prompt engineering - samo numeričke vrednosti
    if 'prompt_engineering' in survey_df.columns:
        # Konvertuj u numeričke vrednosti i filtiraj nevalidne
        numeric_prompt_skills = pd.to_numeric(survey_df['prompt_engineering'], errors='coerce')
        valid_prompt_skills = numeric_prompt_skills.dropna()
        if not valid_prompt_skills.empty:
            prompt_skills = valid_prompt_skills.value_counts().sort_index()
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
