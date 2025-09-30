"""
Data Science Extension za Anketa Sistem
Kreira povezane CSV fajlove za napredne analize
"""

import pandas as pd
import numpy as np
import networkx as nx
import csv
import os
from datetime import datetime
import uuid

class DataScienceManager:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.ensure_directories()
        
        # Definisanje fajlova
        self.files = {
            'participants': f'{base_dir}/participants.csv',
            'ai_knowledge': f'{base_dir}/ai_knowledge.csv', 
            'quiz_responses': f'{base_dir}/quiz_responses.csv',
            'relationships': f'{base_dir}/relationships.csv',
            'tool_usage_patterns': f'{base_dir}/tool_usage_patterns.csv',
            'demographic_clusters': f'{base_dir}/demographic_clusters.csv'
        }
        
        self.init_csv_files()
    
    def ensure_directories(self):
        """Kreiraj direktorijume ako ne postoje"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
    
    def init_csv_files(self):
        """Inicijalizuj CSV fajlove sa headerima"""
        
        # participants.csv
        if not os.path.exists(self.files['participants']):
            with open(self.files['participants'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'timestamp', 'birth_year', 'country', 'education',
                    'employment_status', 'ciljana_grupa', 'work_experience', 'institution', 'field',
                    'programming_experience', 'age_group', 'experience_level'
                ])
        
        # ai_knowledge.csv  
        if not os.path.exists(self.files['ai_knowledge']):
            with open(self.files['ai_knowledge'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'ciljana_grupa', 'ai_tool', 'knowledge_level', 'usage_frequency',
                    'purpose', 'effectiveness_rating', 'tool_category'
                ])
        
        # quiz_responses.csv
        if not os.path.exists(self.files['quiz_responses']):
            with open(self.files['quiz_responses'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'ciljana_grupa', 'question_id', 'question_category', 'given_answer', 
                    'correct_answer', 'is_correct', 'difficulty_level'
                ])
        
        # relationships.csv (za network analizu)
        if not os.path.exists(self.files['relationships']):
            with open(self.files['relationships'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'source_id', 'target_id', 'relationship_type', 'weight', 'context', 'ciljana_grupa'
                ])
        
        # tool_usage_patterns.csv
        if not os.path.exists(self.files['tool_usage_patterns']):
            with open(self.files['tool_usage_patterns'], 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'participant_id', 'ciljana_grupa', 'usage_pattern', 'tools_combination', 'frequency_score',
                    'effectiveness_score', 'pattern_type'
                ])
    
    def process_existing_survey_data(self, survey_csv_path="survey_responses.csv"):
        """Obradi postojeći survey_responses.csv fajl"""
        if not os.path.exists(survey_csv_path):
            print(f"❌ Fajl {survey_csv_path} ne postoji!")
            return 0
            
        # Očisti postojeće data science CSV fajlove
        self.clear_data_files()
        self.init_csv_files()
        
        processed_count = 0
        
        with open(survey_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['timestamp']:  # Proverava da nije prazan red
                    participant_id = self.process_survey_response(row, from_existing=True)
                    processed_count += 1
                    # Ovde koristiti printf samo u procesu testiranja
                    # print(f" Obrađen učesnik {processed_count}: {participant_id}")
        # I ovu liniju koristiti samo u testiranju
        # print(f" Uspešno obrađeno {processed_count} postojećih odgovora!")
        return processed_count
    
    def clear_data_files(self):
        """Očisti postojeće data science fajlove"""
        for file_path in self.files.values():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except PermissionError:
                    print(f"Nije moguće obrisati {file_path} - fajl je u upotrebi !")
                    
                    try:
                        with open(file_path, 'w', newline='', encoding='utf-8') as f:
                            f.write('')
                    except:
                        pass

    def process_survey_response(self, survey_data, from_existing=False):
        """Obradi jedan odgovor iz ankete i razdeli ga po CSV fajlovima"""
        
        # Generiši jedinstveni ID za učesnika
        if from_existing:
            # Za postojeće podatke, generiši ID baziran na timestamp-u za konzistentnost
            timestamp_part = survey_data.get('timestamp', '')[:16].replace(' ', '').replace('-', '').replace(':', '')
            participant_id = f"P{hash(timestamp_part) % 100000000:08X}"
        else:
            participant_id = f"P{str(uuid.uuid4())[:8].upper()}"
        
        # 1. Sacuvaj osnovne demografske podatke
        self.save_participant_data(participant_id, survey_data)
        
        # 2. Obradi AI znanje
        self.process_ai_knowledge(participant_id, survey_data)
        
        # 3. Obradi kviz odgovore
        self.process_quiz_responses(participant_id, survey_data)
        
        # 4. Kreiraj veze (relationships)
        self.create_relationships(participant_id, survey_data)
        
        # 5. Analiziraj patterns korišćenja
        self.analyze_usage_patterns(participant_id, survey_data)
        
        return participant_id
    
    def save_participant_data(self, participant_id, data):
        """Sacuvaj osnovne podatke o učesniku"""
        age_group = self.categorize_age(data.get('godina_rodjenja', ''))
        exp_level = self.categorize_experience(data.get('godine_staza', ''))
        
        participant_row = [
            participant_id,
            data.get('timestamp', ''),
            data.get('godina_rodjenja', ''),
            data.get('drzava', ''),
            data.get('strucna_sprema', ''),
            data.get('radni_odnos', ''),
            data.get('ciljana_grupa', ''),  # Dodano ciljana_grupa
            data.get('godine_staza', ''),
            data.get('institucija', ''),
            data.get('grana_oblast', ''),
            data.get('pisanje_softvera', ''),
            age_group,
            exp_level
        ]
        
        with open(self.files['participants'], 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(participant_row)
    
    def process_ai_knowledge(self, participant_id, data):
        """Obradi podatke o AI znanju"""
        # AI alati koje poznaje
        known_tools = data.get('poznati_ai_alati', '').split(',')
        ai_knowledge_level = data.get('generativni_ai_poznavanje', '')
        ciljana_grupa = data.get('ciljana_grupa', '')
        
        for tool in known_tools:
            if tool.strip():
                tool_category = self.categorize_ai_tool(tool.strip())
                
                # Svrhe korišćenja
                purposes = data.get('svrhe_koriscenja', '').split(',')
                for purpose in purposes:
                    if purpose.strip():
                        knowledge_row = [
                            participant_id,
                            ciljana_grupa,  # Dodano ciljana_grupa
                            tool.strip(),
                            ai_knowledge_level,
                            'unknown',  # frequency - možete dodati polje u anketu
                            purpose.strip(),
                            ai_knowledge_level,  # effectiveness rating
                            tool_category
                        ]
                        
                        with open(self.files['ai_knowledge'], 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(knowledge_row)
    
    def process_quiz_responses(self, participant_id, data):
        """Obradi odgovore na kviz pitanja"""
        ciljana_grupa = data.get('ciljana_grupa', '')
        
        quiz_questions = {
            'chatgpt_omni': {'correct': 'GPT-4', 'category': 'technical', 'difficulty': 'medium'},
            'copilot_task': {'correct': 'Copilot Workspace', 'category': 'technical', 'difficulty': 'hard'},
            'copilot_chat': {'correct': 'Copilot X', 'category': 'technical', 'difficulty': 'medium'},
            'google_model': {'correct': 'Gemini', 'category': 'knowledge', 'difficulty': 'easy'},
            'gpt_realtime': {'correct': 'GPT-4', 'category': 'technical', 'difficulty': 'hard'},
            'codex_successor': {'correct': 'GPT-3.5', 'category': 'technical', 'difficulty': 'medium'},
            'chatgpt_data_analysis': {'correct': 'Advanced Data Analysis (Code Interpreter)', 'category': 'technical', 'difficulty': 'hard'},
            'copilot_workspace': {'correct': 'Copilot Workspace', 'category': 'technical', 'difficulty': 'hard'},
            'anthropic_model': {'correct': 'Claude', 'category': 'knowledge', 'difficulty': 'easy'},
            'creativity_parameter': {'correct': 'Temperature', 'category': 'conceptual', 'difficulty': 'hard'},
            'transformer_basis': {'correct': 'Transformeri', 'category': 'conceptual', 'difficulty': 'medium'},
            'university_guidelines': {'correct': 'Stanford', 'category': 'knowledge', 'difficulty': 'easy'}
        }
        
        for question_id, question_info in quiz_questions.items():
            given_answer = data.get(question_id, '')
            if given_answer:
                is_correct = given_answer == question_info['correct']
                
                quiz_row = [
                    participant_id,
                    ciljana_grupa,  # Dodano ciljana_grupa
                    question_id,
                    question_info['category'],
                    given_answer,
                    question_info['correct'],
                    is_correct,
                    question_info['difficulty']
                ]
                
                with open(self.files['quiz_responses'], 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(quiz_row)
    
    def create_relationships(self, participant_id, data):
        """Kreiraj veze za network analizu"""
        
        ciljana_grupa = data.get('ciljana_grupa', '')
        
        # Mapiranje tekstualnih vrednosti u numeričke
        knowledge_mapping = {
            'nimalo': 1,
            'povrsno': 2, 
            'umereno': 3,
            'dobro': 4,
            'veoma_dobro': 5,
            '': 1  # default vrednost
        }
        
        # Veza između učesnika i ciljane grupe (najvažnija veza za community detection)
        if ciljana_grupa and ciljana_grupa.strip():
            relationship_row = [
                participant_id,
                f"GROUP_{ciljana_grupa.upper()}",
                'belongs_to_group',
                1.0,  # Maksimalna težina za grupu
                'target_group',
                ciljana_grupa
            ]
            
            with open(self.files['relationships'], 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(relationship_row)
        
        # Veza između učesnika i AI alata
        known_tools = data.get('poznati_ai_alati', '').split(',')
        for tool in known_tools:
            if tool.strip():
                # Mapiranje znanja u numeričku vrednost
                knowledge_str = data.get('generativni_ai_poznavanje', 'nimalo').strip()
                knowledge_level = knowledge_mapping.get(knowledge_str, 1)
                weight = float(knowledge_level) / 5.0
                
                relationship_row = [
                    participant_id,
                    f"TOOL_{tool.strip().replace(' ', '_')}",
                    'uses_tool',
                    weight,
                    'ai_tool_usage',
                    ciljana_grupa
                ]
                
                with open(self.files['relationships'], 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(relationship_row)
        
        # Veze sa institutionama
        institution = data.get('institucija', '')
        if institution and institution.strip() and institution != '","':
            relationship_row = [
                participant_id,
                f"ORG_{institution.replace(' ', '_').replace(',', '')}",
                'affiliated_with',
                1.0,
                'organizational',
                ciljana_grupa
            ]
            
            with open(self.files['relationships'], 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(relationship_row)
    
    def analyze_usage_patterns(self, participant_id, data):
        """Analiziraj patterns korišćenja alata"""
        ciljana_grupa = data.get('ciljana_grupa', '')
        known_tools = data.get('poznati_ai_alati', '').split(',')
        purposes = data.get('svrhe_koriscenja', '').split(',')
        
        # Kombinuj alate i svrhe da kreiraš pattern
        tools_clean = [t.strip() for t in known_tools if t.strip()]
        purposes_clean = [p.strip() for p in purposes if p.strip()]
        
        if tools_clean and purposes_clean:
            pattern = f"{len(tools_clean)}_tools_for_{len(purposes_clean)}_purposes"
            tools_combination = "|".join(sorted(tools_clean))
            
            frequency_score = min(len(tools_clean) * len(purposes_clean), 10) / 10.0
            
            # Mapiranje znanja u numeričku vrednost 
            knowledge_mapping = {
                'nimalo': 1,
                'povrsno': 2, 
                'umereno': 3,
                'dobro': 4,
                'veoma_dobro': 5,
                '': 1  # default vrednost
            }
            
            knowledge_str = data.get('generativni_ai_poznavanje', 'nimalo').strip()
            knowledge_level = knowledge_mapping.get(knowledge_str, 1)
            effectiveness_score = float(knowledge_level) / 5.0
            
            pattern_type = self.classify_usage_pattern(len(tools_clean), len(purposes_clean))
            
            pattern_row = [
                participant_id,
                ciljana_grupa,  # Dodano ciljana_grupa
                pattern,
                tools_combination,
                frequency_score,
                effectiveness_score,
                pattern_type
            ]
            
            with open(self.files['tool_usage_patterns'], 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(pattern_row)
    
    # Helper functions
    def categorize_age(self, birth_year):
        if not birth_year or not birth_year.isdigit():
            return 'unknown'
        
        age = 2025 - int(birth_year)
        if age < 25:
            return 'young'
        elif age < 35:
            return 'mid_career'
        elif age < 50:
            return 'experienced' 
        else:
            return 'senior'
    
    def categorize_experience(self, years):
        if not years:
            return 'unknown'
            
        # Mapiranje tekstualnih vrednosti
        if isinstance(years, str):
            years = years.strip()
            if years == '0':
                return 'junior'
            elif years in ['1-2', '1', '2']:
                return 'junior' 
            elif years in ['3-5', '3', '4', '5']:
                return 'intermediate'
            elif years in ['6-10', '6', '7', '8', '9', '10']:
                return 'senior'
            elif 'više' in years.lower() or '10+' in years:
                return 'expert'
            else:
                return 'unknown'
        
        # Ako je numerička vrednost        
        try:
            exp = float(years)
            if exp < 2:
                return 'junior'
            elif exp < 5:
                return 'intermediate'
            elif exp < 10:
                return 'senior'
            else:
                return 'expert'
        except:
            return 'unknown'
    
    def categorize_ai_tool(self, tool):
        tool_lower = tool.lower()
        if 'chatgpt' in tool_lower or 'gpt' in tool_lower:
            return 'conversational_ai'
        elif 'copilot' in tool_lower:
            return 'code_assistant'
        elif 'claude' in tool_lower:
            return 'conversational_ai'
        elif 'gemini' in tool_lower or 'bard' in tool_lower:
            return 'conversational_ai'
        else:
            return 'other'
    
    def classify_usage_pattern(self, num_tools, num_purposes): 
        if num_tools == 1 and num_purposes == 1:
            return 'focused_user'
        elif num_tools > 3 and num_purposes > 3:
            return 'power_user'
        elif num_tools > num_purposes:
            return 'tool_explorer'
        elif num_purposes > num_tools:
            return 'purpose_diverse'
        else:
            return 'balanced_user'
    
    # Network Analysis Methods
    def create_network_graph(self):
        """Kreiraj NetworkX graf za analizu"""
        G = nx.Graph()
        
        # Učitaj relationships sa eksplicitnim dtype za ID kolone
        try:
            if not os.path.exists(self.files['relationships']):
                print(" File relationships.csv ne postoji !")
                return G
                
            # Pročitaj fajl i provjeri da li ima header
            with open(self.files['relationships'], 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            # Ako prvi red ne sadrži 'source_id', fajl nema header
            if 'source_id' not in first_line:
                print("File relationships.csv nema header - dodajemo header ...")
                # Čitaj podatke bez header-a
                relationships_df = pd.read_csv(self.files['relationships'], 
                                             header=None,
                                             names=['source_id', 'target_id', 'relationship_type', 
                                                   'weight', 'context', 'ciljana_grupa'],
                                             dtype={'source_id': str, 'target_id': str},
                                             encoding='utf-8')
            else:
                relationships_df = pd.read_csv(self.files['relationships'], 
                                             dtype={'source_id': str, 'target_id': str},
                                             encoding='utf-8')
                                             
        except Exception as e:
            print(f"Greška pri čitanju relationships fajla: {e}")
            return G
        
        for _, row in relationships_df.iterrows():
            try:
                # Eksplicitno konvertuj u string da izbegneš probleme sa tipovima
                source_id = str(row['source_id'])
                target_id = str(row['target_id'])
                
                G.add_edge(
                    source_id, 
                    target_id,
                    relationship_type=row.get('relationship_type', 'unknown'),
                    weight=float(row.get('weight', 1.0)),
                    context=row.get('context', 'unknown'),
                    ciljana_grupa=row.get('ciljana_grupa', 'unknown')
                )
            except Exception as e:
                print(f"Greška pri dodavanju veze {row}: {e}")
                continue
        
        return G
    
    def generate_gephi_files(self):
        """Generiši fajlove za Gephi sa ciljanim grupama kao komunama"""
        try:
            G = self.create_network_graph()
            print(f"Network graf kreiran sa {G.number_of_nodes()} čvorova i {G.number_of_edges()} veza")
            
            if G.number_of_nodes() == 0:
                print("Graf nema čvorove - proverite relationships.csv fajl !")
                return None, None
            
            # Učitaj dodatne informacije o učesnicima
            try:
                participants_df = pd.read_csv(self.files['participants'], encoding='utf-8')
                # Ukloni duplikate na osnovu participant_id i uzmi poslednji unos
                participants_df = participants_df.drop_duplicates(subset=['participant_id'], keep='last')
                participant_info = participants_df.set_index('participant_id').to_dict('index')
                print(f"Učitano {len(participant_info)} učesnika iz participants.csv (nakon uklanjanja duplikata)")
            except Exception as e:
                print(f"Greška pri čitanju participants.csv: {e}")
                participant_info = {}
            
            # Nodes file sa dodatnim atributima za ciljane grupe
            nodes_data = []
            for node in G.nodes():
                node_str = str(node)
                
                # Određuj tip čvora
                if node_str.startswith('P'):
                    node_type = 'participant'
                    # Dobij informacije o ciljanoj grupi
                    participant_data = participant_info.get(node_str, {})
                    ciljana_grupa = participant_data.get('ciljana_grupa', 'unknown')
                    age_group = participant_data.get('age_group', 'unknown')
                    exp_level = participant_data.get('experience_level', 'unknown')
                    
                    nodes_data.append({
                        'Id': node_str,
                        'Label': node_str,
                        'Type': node_type,
                        'CiljanaGrupa': ciljana_grupa,
                        'AgeGroup': age_group,
                        'ExperienceLevel': exp_level,
                        'Size': 20,
                        'Color': self.get_group_color(ciljana_grupa)
                    })
                    
                elif node_str.startswith('GROUP_'):
                    node_type = 'target_group'
                    group_name = node_str.replace('GROUP_', '')
                    
                    nodes_data.append({
                        'Id': node_str,
                        'Label': group_name,
                        'Type': node_type,
                        'CiljanaGrupa': group_name.lower(),
                        'AgeGroup': 'group',
                        'ExperienceLevel': 'group',
                        'Size': 50,  # Veći čvorovi za grupe
                        'Color': self.get_group_color(group_name.lower())
                    })
                    
                elif node_str.startswith('TOOL_'):
                    node_type = 'ai_tool'
                    tool_name = node_str.replace('TOOL_', '').replace('_', ' ')
                    
                    nodes_data.append({
                        'Id': node_str,
                        'Label': tool_name,
                        'Type': node_type,
                        'CiljanaGrupa': 'tool',
                        'AgeGroup': 'tool',
                        'ExperienceLevel': 'tool',
                        'Size': 30,
                        'Color': '#FFD700'  # Zlatna za alate
                    })
                    
                else:
                    node_type = 'entity'
                    nodes_data.append({
                        'Id': node_str,
                        'Label': node_str,
                        'Type': node_type,
                        'CiljanaGrupa': 'other',
                        'AgeGroup': 'other',
                        'ExperienceLevel': 'other',
                        'Size': 15,
                        'Color': '#808080'  # Siva za ostale
                    })
            
            print(f"Kreano {len(nodes_data)} čvorova za Gephi")
            
            if nodes_data:
                nodes_df = pd.DataFrame(nodes_data)
                nodes_file_path = f'{self.base_dir}/gephi_nodes.csv'
                nodes_df.to_csv(nodes_file_path, index=False)
                print(f"Nodes fajl snimljen: {nodes_file_path}")
            
            # Edges file sa dodatnim atributima
            edges_data = []
            for edge in G.edges(data=True):
                relationship_type = edge[2].get('relationship_type', 'unknown')
                weight = edge[2].get('weight', 1.0)
                context = edge[2].get('context', 'unknown')
                
                # Definiši boju veze na osnovu tipa
                edge_color = self.get_edge_color(relationship_type)
                
                edges_data.append({
                    'Source': str(edge[0]),
                    'Target': str(edge[1]),
                    'Type': relationship_type,
                    'Weight': weight,
                    'Context': context,
                    'Color': edge_color,
                    'Thickness': max(1, int(weight * 5))  # Debljina na osnovu težine
                })
            
            print(f"Kreano {len(edges_data)} veza za Gephi")
            
            if edges_data:
                edges_df = pd.DataFrame(edges_data)
                edges_file_path = f'{self.base_dir}/gephi_edges.csv'
                edges_df.to_csv(edges_file_path, index=False)
                print(f"Edges fajl snimljen: {edges_file_path}")
            
            return f'{self.base_dir}/gephi_nodes.csv', f'{self.base_dir}/gephi_edges.csv'
            
        except Exception as e:
            print(f"Greška pri generisanju Gephi fajlova: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def get_group_color(self, group_name):
        """Vrati boju za ciljanu grupu"""
        colors = {
            'it': '#FF6B6B',          # Crvena za IT
            'prosveta': '#4ECDC4',    # Tirkizna za prosvetu
            'medicina': '#45B7D1',    # Plava za medicinu
            'kreativna': '#96CEB4',   # Zelena za kreativne industrije
            'drustvene': '#FFEAA7',   # Žuta za društvene nauke
            'ostalo': '#DDA0DD'       # Ljubičasta za ostalo
        }
        return colors.get(group_name.lower(), '#C0C0C0')  # Default siva
    
    def get_edge_color(self, relationship_type):
        """Vrati boju za tip veze"""
        colors = {
            'belongs_to_group': '#FF0000',    # Crvena za pripadnost grupi
            'uses_tool': '#00FF00',          # Zelena za korišćenje alata
            'affiliated_with': '#0000FF'     # Plava za afilijaciju
        }
        return colors.get(relationship_type, '#808080')  # Default siva
    
    def get_analytics_summary(self):
        """Generiši sažetak za analitiku"""
        summary = {}
        
        try:
            # Učitaj podatke
            participants_df = pd.read_csv(self.files['participants'], encoding='utf-8')
            ai_knowledge_df = pd.read_csv(self.files['ai_knowledge'], encoding='utf-8')
            quiz_df = pd.read_csv(self.files['quiz_responses'], encoding='utf-8')
            patterns_df = pd.read_csv(self.files['tool_usage_patterns'], encoding='utf-8')
            
            summary = {
                'total_participants': len(participants_df),
                'total_ai_interactions': len(ai_knowledge_df),
                'total_quiz_responses': len(quiz_df),
                'unique_tools': ai_knowledge_df['ai_tool'].nunique(),
                'avg_quiz_score': quiz_df['is_correct'].mean() * 100,
                'pattern_types': patterns_df['pattern_type'].value_counts().to_dict(),
                'most_popular_tools': ai_knowledge_df['ai_tool'].value_counts().head(5).to_dict(),
                'files_created': list(self.files.keys())
            }
            
        except Exception as e:
            summary = {'error': str(e)}
        
        return summary

# Example usage functions
def integrate_with_flask_app():
    """Primer integracije sa Flask aplikacijom"""
    
    # U submit_survey funkciji dodaj:
    """
    data_manager = DataScienceManager()
    participant_id = data_manager.process_survey_response(data)
    
    # Dodaj participant_id u session za tracking
    session['participant_id'] = participant_id
    """
    
def generate_advanced_analytics():
    """Primer naprednih analiza"""
    data_manager = DataScienceManager()
    
    # NetworkX analiza
    G = data_manager.create_network_graph()
    
    # Centralnost mera
    centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    # Community detection
    communities = nx.community.louvain_communities(G)
    
    # Gephi export
    nodes_file, edges_file = data_manager.generate_gephi_files()
    
    print(f"Created Gephi files: {nodes_file}, {edges_file}")
    print(f"Found {len(communities)} communities")
    print(f"Top central nodes: {sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]}")

if __name__ == "__main__":
    # Test the system
    data_manager = DataScienceManager()
    summary = data_manager.get_analytics_summary()
    print("Data Science System Summary:")
    print(summary)
