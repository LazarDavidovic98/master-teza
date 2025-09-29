"""
Napredne Data Science Analize
Demonstrira mogućnosti analiza sa povezanim CSV fajlovima
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.load_all_data()
    
    def load_all_data(self):
        """Učitaj sve CSV fajlove"""
        try:
            self.participants = pd.read_csv(f'{self.data_dir}/participants.csv')
            self.ai_knowledge = pd.read_csv(f'{self.data_dir}/ai_knowledge.csv') 
            self.quiz_responses = pd.read_csv(f'{self.data_dir}/quiz_responses.csv')
            
            # Za relationships.csv, pokušaj prvo sa headerom, zatim bez
            try:
                self.relationships = pd.read_csv(f'{self.data_dir}/relationships.csv')
                # Proveriti da li ima potrebne kolone
                if 'source_id' not in self.relationships.columns:
                    raise ValueError("Nema source_id kolonu")
            except:
                # Pokušaj učitati bez header-a
                self.relationships = pd.read_csv(f'{self.data_dir}/relationships.csv', 
                                               header=None,
                                               names=['source_id', 'target_id', 'relationship_type', 'weight', 'category', 'group'])
            
            self.usage_patterns = pd.read_csv(f'{self.data_dir}/tool_usage_patterns.csv')
            
        except FileNotFoundError as e:
            print(f"Fajl nije pronađen: {e}")
            print("Prvo pokrenite data_science_extension.py da kreirate podatke")
    
    def demographic_analysis(self):
        """Demografska analiza sa cross-tabulation"""
        print("=== DEMOGRAFSKA ANALIZA ===")
        
        # Age vs Education correlation
        age_edu = pd.crosstab(self.participants['age_group'], 
                             self.participants['education'])
        print("\\nAge Group vs Education:")
        print(age_edu)
        
        # Employment vs Experience
        emp_exp = pd.crosstab(self.participants['employment_status'],
                             self.participants['experience_level'])
        print("\\nEmployment vs Experience Level:")
        print(emp_exp)
        
        return age_edu, emp_exp
    
    def ai_tool_clustering(self):
        """Klasterovanje korisnika na osnovu AI alata"""
        print("\\n=== AI TOOL CLUSTERING ===")
        
        # Proveriti da li postoji knowledge_level kolona i da li ima validne podatke
        if 'knowledge_level' not in self.ai_knowledge.columns:
            print("UPOZORENJE: knowledge_level kolona ne postoji u ai_knowledge.csv")
            return None, None
        
        # Proveriti da li su knowledge_level podaci validni
        valid_knowledge = self.ai_knowledge['knowledge_level'].notna() & \
                         (self.ai_knowledge['knowledge_level'] != '') & \
                         (self.ai_knowledge['knowledge_level'] != 'unknown')
        
        if not valid_knowledge.any():
            print("UPOZORENJE: Nema validnih knowledge_level podataka")
            print("Koristiću alternativnu analizu na osnovu broja AI alata po korisniku...")
            
            # Alternativna analiza - broji koliko svaki korisnik koristi različitih alata
            try:
                tool_counts = self.ai_knowledge.groupby('participant_id')['ai_tool'].nunique().reset_index()
                tool_counts.columns = ['participant_id', 'tool_count']
                
                # Kreiraj dummy pivot tabelu - koristi ai_tool kao index umesto participant_id
                tool_matrix = self.ai_knowledge.pivot_table(
                    index='participant_id', 
                    columns='ai_tool', 
                    aggfunc='size',  # Koristi size umesto count
                    fill_value=0
                )
                
                print(f"Analiza na osnovu {len(tool_matrix.columns)} različitih AI alata")
                print(f"Korisnici u analizi: {len(tool_matrix)}")
                
                # Proveri da li imamo podatke za clustering
                if tool_matrix.empty or len(tool_matrix) < 2:
                    print("GREŠKA: Nedovoljno podataka za clustering analizu")
                    return None, None
                
                # Standardizuj podatke
                scaler = StandardScaler()
                tool_matrix_scaled = scaler.fit_transform(tool_matrix.values)
                
                # K-means clustering sa prilagođenim brojem klastera
                n_clusters = min(4, len(tool_matrix) // 2, len(tool_matrix))
                if n_clusters < 2:
                    n_clusters = 2
                    
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tool_matrix_scaled)
                
                # Dodaj cluster labels
                tool_matrix_copy = tool_matrix.copy()
                tool_matrix_copy['cluster'] = clusters
                
                print(f"Kreirano {len(set(clusters))} klastera korisnika")
                print("\\nDistribucija po klasterima:")
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                for cluster, count in cluster_counts.items():
                    print(f"Klaster {cluster}: {count} korisnika")
                
                return tool_matrix_copy, clusters
                
            except Exception as e:
                print(f"Greška u alternativnoj analizi: {e}")
                return None, None
        else:
            # Originalna analiza sa knowledge_level
            filtered_data = self.ai_knowledge[valid_knowledge].copy()
            
            # Pokušaj konvertovati knowledge_level u numeričke vrednosti
            try:
                filtered_data['knowledge_level'] = pd.to_numeric(filtered_data['knowledge_level'])
            except:
                print("UPOZORENJE: Ne mogu da konvertujem knowledge_level u brojeve")
                return None, None
            
            # Kreiraj pivot tabelu: participant vs ai_tool
            tool_matrix = filtered_data.pivot_table(
                index='participant_id', 
                columns='ai_tool', 
                values='knowledge_level', 
                fill_value=0
            )
            
            if tool_matrix.empty:
                print("GREŠKA: Pivot tabela je prazna")
                return None, None
                
            # Standardizuj podatke
            scaler = StandardScaler()
            tool_matrix_scaled = scaler.fit_transform(tool_matrix)
            
            # K-means clustering
            n_clusters = min(4, len(tool_matrix))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(tool_matrix_scaled)
            
            # Dodaj cluster labels
            tool_matrix['cluster'] = clusters
            
            print(f"Kreirano {len(set(clusters))} klastera korisnika")
            print("\\nDistribucija po klasterima:")
            print(pd.Series(clusters).value_counts().sort_index())
            
            return tool_matrix, clusters
    
    def quiz_performance_analysis(self):
        """Analiza performansi na kvizu"""
        print("\\n=== QUIZ PERFORMANCE ANALYSIS ===")
        
        # Performance po kategorijama
        category_performance = self.quiz_responses.groupby('question_category').agg({
            'is_correct': ['mean', 'count']
        }).round(3)
        
        print("\\nPerformanse po kategorijama pitanja:")
        print(category_performance)
        
        # Performance po težini
        difficulty_performance = self.quiz_responses.groupby('difficulty_level').agg({
            'is_correct': ['mean', 'count']
        }).round(3)
        
        print("\\nPerformanse po težini pitanja:")
        print(difficulty_performance)
        
        # Individual participant scores
        participant_scores = self.quiz_responses.groupby('participant_id').agg({
            'is_correct': 'mean'
        }).round(3)
        
        print(f"\\nProsečan skor: {participant_scores['is_correct'].mean():.3f}")
        print(f"Najbolji skor: {participant_scores['is_correct'].max():.3f}")
        print(f"Najgori skor: {participant_scores['is_correct'].min():.3f}")
        
        return category_performance, difficulty_performance, participant_scores
    
    def network_analysis(self):
        """NetworkX analiza veza"""
        print("\\n=== NETWORK ANALYSIS ===")
        
        # Kreiraj graf
        G = nx.Graph()
        
        try:
            for _, row in self.relationships.iterrows():
                if pd.notna(row['source_id']) and pd.notna(row['target_id']) and pd.notna(row['weight']):
                    # Konvertuj weight u float
                    try:
                        weight = float(row['weight'])
                    except:
                        weight = 1.0
                    
                    G.add_edge(str(row['source_id']), str(row['target_id']), 
                              weight=weight,
                              relationship_type=str(row['relationship_type']))
        except Exception as e:
            print(f"Greška pri kreiranju grafa: {e}")
            return None, None, None, None
        
        print(f"Graf ima {G.number_of_nodes()} čvorova i {G.number_of_edges()} veza")
        
        if G.number_of_nodes() == 0:
            print("UPOZORENJE: Graf je prazan")
            return None, None, None, None
        
        # Centralnost mere
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
        except Exception as e:
            print(f"Greška pri računanju centralnosti: {e}")
            return G, None, None, None
        
        # Top 5 centralnijih čvorova
        print("\\nTop 5 čvorova po degree centralnosti:")
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, centrality in top_degree:
            print(f"{node}: {centrality:.3f}")
        
        # Community detection
        try:
            communities = list(nx.community.louvain_communities(G))
            print(f"\\nPronađeno {len(communities)} zajednica u mreži")
            for i, community in enumerate(communities[:3]):  # Prikaži prve 3 zajednice
                print(f"Zajednica {i+1}: {len(community)} čvorova")
        except Exception as e:
            print(f"Greška pri community detection: {e}")
            communities = []
        
        return G, degree_centrality, betweenness_centrality, communities
    
    def correlation_analysis(self):
        """Korelaciona analiza između različitih varijabli"""
        print("\\n=== CORRELATION ANALYSIS ===")
        
        # Merge datasets za korelacije
        # Participants sa quiz performance
        participant_quiz = self.quiz_responses.groupby('participant_id').agg({
            'is_correct': 'mean'
        }).rename(columns={'is_correct': 'quiz_score'})
        
        # Participants sa AI knowledge - korisiti tool count umesto knowledge_level
        # jer je knowledge_level kolona problematična
        participant_ai = self.ai_knowledge.groupby('participant_id').agg({
            'ai_tool': 'nunique'  # Broji broj različitih alata koje korisnik koristi
        }).rename(columns={'ai_tool': 'ai_tools_count'})
        
        # Merge all
        merged_data = self.participants.set_index('participant_id')
        merged_data = merged_data.join([participant_quiz, participant_ai], how='left')
        
        # Popuni NaN vrednosti
        merged_data['quiz_score'] = merged_data['quiz_score'].fillna(0)
        merged_data['ai_tools_count'] = merged_data['ai_tools_count'].fillna(0)
        
        # Konvertuj kategoričke varijable u numeričke
        merged_data['employment_numeric'] = merged_data['employment_status'].map({
            'da': 1, 'ne': 0
        }).fillna(0)
        
        merged_data['education_numeric'] = merged_data['education'].map({
            'srednja': 1, 'visa': 2, 'visoka': 3, 'master': 4, 'doktorat': 5
        }).fillna(1)
        
        # Korelaciona matrica
        numeric_cols = ['quiz_score', 'ai_tools_count', 'employment_numeric', 'education_numeric']
        
        # Proveriti da li postoje potrebne kolone
        available_cols = [col for col in numeric_cols if col in merged_data.columns]
        
        if len(available_cols) < 2:
            print("UPOZORENJE: Nedovoljno numeričkih kolona za korelacionu analizu")
            return None, merged_data
            
        correlation_matrix = merged_data[available_cols].corr()
        
        print("\\nKorelaciona matrica:")
        print(correlation_matrix.round(3))
        
        print("\\nInterpretacija:")
        print(f"• quiz_score: prosečan uspeh na kvizu po korisniku")
        print(f"• ai_tools_count: broj različitih AI alata koje korisnik koristi")
        print(f"• employment_numeric: da li je korisnik zaposlen (1=da, 0=ne)")
        print(f"• education_numeric: nivo obrazovanja (1=srednja, 5=doktorat)")
        
        return correlation_matrix, merged_data
    
    def usage_pattern_analysis(self):
        """Analiza patterns korišćenja"""
        print("\\n=== USAGE PATTERN ANALYSIS ===")
        
        # Distribucija pattern types
        pattern_dist = self.usage_patterns['pattern_type'].value_counts()
        print("\\nDistribucija tipova korisnika:")
        for pattern, count in pattern_dist.items():
            print(f"{pattern}: {count}")
        
        # Effectiveness po pattern type
        effectiveness_by_pattern = self.usage_patterns.groupby('pattern_type').agg({
            'effectiveness_score': ['mean', 'std', 'count']
        }).round(3)
        
        print("\\nEfektivnost po tipovima korisnika:")
        print(effectiveness_by_pattern)
        
        # Most common tool combinations
        top_combinations = self.usage_patterns['tools_combination'].value_counts().head(10)
        print("\\nNajčešće kombinacije alata:")
        for combo, count in top_combinations.items():
            print(f"{combo}: {count}")
        
        return pattern_dist, effectiveness_by_pattern, top_combinations
    
    def generate_insights(self):
        """Generiši ključne insights"""
        print("\\n" + "="*50)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("="*50)
        
        insights = []
        
        # 1. Demographics insight
        if hasattr(self, 'participants'):
            age_groups = self.participants['age_group'].value_counts()
            dominant_age = age_groups.index[0]
            insights.append(f"• Dominantna starosna grupa: {dominant_age} ({age_groups.iloc[0]} učesnika)")
        
        # 2. AI Tool insight
        if hasattr(self, 'ai_knowledge'):
            popular_tool = self.ai_knowledge['ai_tool'].value_counts().index[0]
            tool_count = self.ai_knowledge['ai_tool'].value_counts().iloc[0]
            insights.append(f"• Najpopularniji AI alat: {popular_tool} ({tool_count} korisnika)")
        
        # 3. Quiz insight
        if hasattr(self, 'quiz_responses'):
            avg_score = self.quiz_responses['is_correct'].mean()
            insights.append(f"• Prosečan uspeh na kvizu: {avg_score:.1%}")
            
            hardest_category = self.quiz_responses.groupby('question_category')['is_correct'].mean().idxmin()
            insights.append(f"• Najteža kategorija pitanja: {hardest_category}")
        
        # 4. Pattern insight
        if hasattr(self, 'usage_patterns'):
            dominant_pattern = self.usage_patterns['pattern_type'].value_counts().index[0]
            insights.append(f"• Najčešći tip korisnika: {dominant_pattern}")
        
        for insight in insights:
            print(insight)
        
        print("\\nPREPORUKE ZA DALJU ANALIZU:")
        print("• Koristite Gephi za vizualizaciju network strukture")
        print("• Implementirajte time-series analizu za praćenje trendova")
        print("• Dodajte sentiment analysis za qualitative responses")
        print("• Kreirajte predictive modele za user behavior")
    
    def export_for_external_tools(self):
        """Export podataka za spoljne alate"""
        print("\\n=== EXPORT FOR EXTERNAL TOOLS ===")
        
        # Za R
        export_dir = f"{self.data_dir}/exports"
        import os
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Merged dataset za R/SPSS
        try:
            merged_for_stats = self.participants.merge(
                self.quiz_responses.groupby('participant_id')['is_correct'].mean(),
                left_on='participant_id', right_index=True, how='left'
            )
            
            merged_for_stats.to_csv(f'{export_dir}/merged_for_r_analysis.csv', index=False)
            print(f"• Kreiran merged dataset: {export_dir}/merged_for_r_analysis.csv")
        except Exception as e:
            print(f"Greška pri kreiranju merged dataseta: {e}")
        
        # Network za Gephi
        try:
            gephi_nodes = []
            gephi_edges = []
            
            # Nodes
            for participant in self.participants['participant_id']:
                gephi_nodes.append({
                    'Id': participant,
                    'Label': participant,
                    'Type': 'participant'
                })
            
            for tool in self.ai_knowledge['ai_tool'].unique():
                gephi_nodes.append({
                    'Id': f"TOOL_{tool}",
                    'Label': tool,
                    'Type': 'ai_tool'
                })
            
            # Edges - koristi podatke iz relationships.csv-a ako su dostupni
            if hasattr(self, 'relationships') and not self.relationships.empty:
                # Proveri da li relationships ima potrebne kolone
                if 'source_id' in self.relationships.columns:
                    for _, row in self.relationships.iterrows():
                        if pd.notna(row['source_id']) and pd.notna(row['target_id']):
                            try:
                                weight = float(row['weight']) if pd.notna(row['weight']) else 1.0
                            except:
                                weight = 1.0
                                
                            gephi_edges.append({
                                'Source': str(row['source_id']),
                                'Target': str(row['target_id']),
                                'Weight': weight,
                                'Type': str(row['relationship_type']) if pd.notna(row['relationship_type']) else 'unknown'
                            })
                else:
                    # Ako nema pravilan header, kreiraj edges iz AI knowledge podataka
                    for _, row in self.ai_knowledge.iterrows():
                        gephi_edges.append({
                            'Source': row['participant_id'],
                            'Target': f"TOOL_{row['ai_tool']}",
                            'Weight': 1.0,
                            'Type': 'uses_tool'
                        })
            else:
                # Fallback - kreiraj edges iz AI knowledge podataka
                for _, row in self.ai_knowledge.iterrows():
                    gephi_edges.append({
                        'Source': row['participant_id'],
                        'Target': f"TOOL_{row['ai_tool']}",
                        'Weight': 1.0,
                        'Type': 'uses_tool'
                    })
            
            pd.DataFrame(gephi_nodes).to_csv(f'{export_dir}/gephi_nodes.csv', index=False)
            pd.DataFrame(gephi_edges).to_csv(f'{export_dir}/gephi_edges.csv', index=False)
            
            print(f"• Kreirani Gephi fajlovi: {export_dir}/gephi_nodes.csv, {export_dir}/gephi_edges.csv")
            
        except Exception as e:
            print(f"Greška pri kreiranju Gephi fajlova: {e}")
        
        return export_dir

def main():
    """Pokretanje kompletne analize"""
    print("POKRETANJE NAPREDNE DATA SCIENCE ANALIZE")
    print("="*60)
    
    # Inicijalizuj
    analytics = AdvancedAnalytics()
    
    try:
        # Pokreni sve analize sa error handling
        print("\\n1. Pokretanje demografske analize...")
        analytics.demographic_analysis()
        
        print("\\n2. Pokretanje AI tool clustering analize...")
        try:
            analytics.ai_tool_clustering() 
        except Exception as e:
            print(f"Greška u AI tool clustering: {e}")
            print("Nastavljam sa ostalim analizama...")
        
        print("\\n3. Pokretanje quiz performance analize...")
        analytics.quiz_performance_analysis()
        
        print("\\n4. Pokretanje network analize...")
        try:
            analytics.network_analysis()
        except Exception as e:
            print(f"Greška u network analizi: {e}")
            print("Nastavljam sa ostalim analizama...")
        
        print("\\n5. Pokretanje korelacione analize...")
        try:
            analytics.correlation_analysis()
        except Exception as e:
            print(f"Greška u korelacionoj analizi: {e}")
            print("Nastavljam sa ostalim analizama...")
        
        print("\\n6. Pokretanje usage pattern analize...")
        try:
            analytics.usage_pattern_analysis()
        except Exception as e:
            print(f"Greška u usage pattern analizi: {e}")
            print("Nastavljam sa ostalim analizama...")
        
        # Generiši insights
        analytics.generate_insights()
        
        # Export za spoljne alate
        try:
            analytics.export_for_external_tools()
        except Exception as e:
            print(f"Greška pri eksportu: {e}")
        
        print("\\n" + "="*60)
        print("ANALIZA ZAVRŠENA!")
        print("Neki moduli možda imaju probleme zbog kvaliteta podataka u CSV fajlovima.")
        print("Proverite 'data/exports/' direktorijum za fajlove za spoljne alate.")
        
    except Exception as e:
        print(f"Kritična greška tokom analize: {e}")
        print("Proverite da li postoje CSV fajlovi u 'data/' direktorijumu")
        print("i da li su u ispravnom formatu.")

if __name__ == "__main__":
    main()
