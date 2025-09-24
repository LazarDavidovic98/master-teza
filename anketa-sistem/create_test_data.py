import csv
from datetime import datetime, timedelta
import random

# Sample test data for the new survey structure
test_data = [
    ['2025-09-24 10:00:00', '1995', 'srbija', 'visoka', 'da', 'informatika', '3-5', 'FTN', 'Web development', '5', '5', '4', '4', 'da', 'da', 'da', 'Claude, Gemini', 'dnevno', 'nedeljno', 'kodiranje,debagovanje', 'boilerplate,refaktorisanje', 'ponekad'],
    ['2025-09-24 11:30:00', '1998', 'crna_gora', 'master', 'da', 'informatika', '1-2', 'PMF', 'DevOps', '4', '5', '3', '5', 'da', 'da', 'da', 'Bard, LLaMA', 'nedeljno', 'dnevno', 'ucenje,kodiranje', 'skelet,dokumentaciju', 'nikad'],
    ['2025-09-24 14:15:00', '2000', 'bosna_i_hercegovina', 'visoka', 'ne', 'student', '0', 'Univerzitet', 'Machine Learning', '3', '4', '2', '3', 'da', 'ne', 'ne', '', 'mesecno', 'nikada', 'ucenje,istrazivanje', '', 'nikad'],
    ['2025-09-24 16:45:00', '1993', 'srbija', 'master', 'da', 'elektrotehnika', '6-10', 'RT-RK', 'Embedded systems', '4', '4', '5', '3', 'da', 'da', 'da', 'Claude, Mistral', 'dnevno', 'mesecno', 'kodiranje,debagovanje', 'testove,refaktorisanje', 'uvek'],
    ['2025-09-24 18:20:00', '1997', 'hrvatska', 'visoka', 'da', 'informatika', '3-5', 'Microsoft', 'Frontend development', '5', '5', '4', '4', 'da', 'da', 'da', 'Gemini, Perplexity', 'svakih_par_sati', 'dnevno', 'kodiranje,pisanje', 'boilerplate,skelet', 'ponekad'],
    ['2025-09-25 09:10:00', '2001', 'slovenija', 'srednja', 'ne', 'student', '0', 'FON', 'Data Science', '2', '3', '1', '2', 'da', 'ne', 'da', 'Claude', 'mesecno', 'nikada', 'ucenje,istrazivanje', '', 'nikad'],
    ['2025-09-25 12:30:00', '1994', 'severna_makedonija', 'doktorat', 'da', 'medicina', '11-15', 'Klinicki centar', 'Bioinformatika', '3', '4', '3', '4', 'da', 'da', 'da', 'Bard, Claude, GPT-4', 'nedeljno', 'mesecno', 'istrazivanje,pisanje', 'dokumentaciju,testove', 'ponekad'],
    ['2025-09-25 15:45:00', '1999', 'srbija', 'visoka', 'da', 'ekonomija', '1-2', 'NBS', 'Financial analysis', '2', '2', '2', '3', 'da', 'ne', 'ne', '', 'nedeljno', 'nikada', 'istrazivanje,pisanje', '', 'nikad'],
    ['2025-09-25 17:20:00', '1996', 'bugarska', 'master', 'da', 'masinstvo', '6-10', 'Siemens', 'Automation', '4', '3', '4', '3', 'da', 'da', 'da', 'Claude, Llama', 'mesecno', 'nedeljno', 'kodiranje,administracija', 'boilerplate,dokumentaciju', 'ponekad'],
    ['2025-09-25 19:00:00', '2002', 'rumunija', 'visa', 'ne', 'student', '0', 'Politehnika', 'Software engineering', '4', '4', '3', '4', 'da', 'da', 'da', 'Gemini, Perplexity', 'dnevno', 'nedeljno', 'ucenje,kodiranje', 'skelet,refaktorisanje', 'nikad']
]

# Write test data to CSV
with open('survey_responses.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow([
        'timestamp', 'godina_rodjenja', 'drzava', 'strucna_sprema', 'radni_odnos',
        'grana_oblast', 'godine_staza', 'institucija', 'uza_oblast',
        'programska_okruzenja', 'programski_jezici', 'pisanje_softvera', 'generativni_ai',
        'cuo_chatgpt', 'cuo_copilot', 'zna_druge_llm', 'navedi_llm',
        'chatgpt_cesto', 'copilot_cesto', 'svrha_koriscenja', 'copilot_svrha', 'copilot_licence'
    ])
    # Write test data
    for row in test_data:
        writer.writerow(row)

print("Test data created successfully!")
print(f"Created {len(test_data)} test records")
