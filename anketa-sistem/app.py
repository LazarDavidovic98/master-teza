from flask import Flask, render_template, request, redirect, url_for, flash
import csv
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# File path for storing survey responses
RESPONSES_FILE = 'survey_responses.csv'

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
