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
            self.relationships = pd.read_csv(f'{self.data_dir}/relationships.csv')
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
        
        # Kreiraj pivot tabelu: participant vs ai_tool
        tool_matrix = self.ai_knowledge.pivot_table(
            index='participant_id', 
            columns='ai_tool', 
            values='knowledge_level', 
            fill_value=0
        )
        
        # Standardizuj podatke
        scaler = StandardScaler()
        tool_matrix_scaled = scaler.fit_transform(tool_matrix)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
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
        
        for _, row in self.relationships.iterrows():
            G.add_edge(row['source_id'], row['target_id'], 
                      weight=row['weight'],
                      relationship_type=row['relationship_type'])
        
        print(f"Graf ima {G.number_of_nodes()} čvorova i {G.number_of_edges()} veza")
        
        # Centralnost mere
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Top 5 centralnijih čvorova
        print("\\nTop 5 čvorova po degree centralnosti:")
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, centrality in top_degree:
            print(f"{node}: {centrality:.3f}")
        
        # Community detection
        communities = list(nx.community.louvain_communities(G))
        print(f"\\nPronađeno {len(communities)} zajednica u mreži")
        
        return G, degree_centrality, betweenness_centrality, communities
    
    def correlation_analysis(self):
        """Korelaciona analiza između različitih varijabli"""
        print("\\n=== CORRELATION ANALYSIS ===")
        
        # Merge datasets za korelacije
        # Participants sa quiz performance
        participant_quiz = self.quiz_responses.groupby('participant_id').agg({
            'is_correct': 'mean'
        }).rename(columns={'is_correct': 'quiz_score'})
        
        # Participants sa AI knowledge
        participant_ai = self.ai_knowledge.groupby('participant_id').agg({
            'knowledge_level': 'mean'
        }).rename(columns={'knowledge_level': 'avg_ai_knowledge'})
        
        # Merge all
        merged_data = self.participants.set_index('participant_id')
        merged_data = merged_data.join([participant_quiz, participant_ai], how='left')
        
        # Konvertuj kategoričke varijable u numeričke
        merged_data['employment_numeric'] = merged_data['employment_status'].map({
            'employed': 1, 'unemployed': 0
        })
        
        merged_data['education_numeric'] = merged_data['education'].map({
            'high_school': 1, 'bachelor': 2, 'master': 3, 'phd': 4
        })
        
        # Korelaciona matrica
        numeric_cols = ['quiz_score', 'avg_ai_knowledge', 'employment_numeric', 'education_numeric']
        correlation_matrix = merged_data[numeric_cols].corr()
        
        print("\\nKorelaciona matrica:")
        print(correlation_matrix.round(3))
        
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
        merged_for_stats = self.participants.merge(
            self.quiz_responses.groupby('participant_id')['is_correct'].mean(),
            left_on='participant_id', right_index=True, how='left'
        )
        
        merged_for_stats.to_csv(f'{export_dir}/merged_for_r_analysis.csv', index=False)
        print(f"• Kreiran merged dataset: {export_dir}/merged_for_r_analysis.csv")
        
        # Network za Gephi
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
        
        # Edges
        for _, row in self.relationships.iterrows():
            gephi_edges.append({
                'Source': row['source_id'],
                'Target': row['target_id'],
                'Weight': row['weight'],
                'Type': row['relationship_type']
            })
        
        pd.DataFrame(gephi_nodes).to_csv(f'{export_dir}/gephi_nodes.csv', index=False)
        pd.DataFrame(gephi_edges).to_csv(f'{export_dir}/gephi_edges.csv', index=False)
        
        print(f"• Kreirani Gephi fajlovi: {export_dir}/gephi_nodes.csv, {export_dir}/gephi_edges.csv")
        
        return export_dir

def main():
    """Pokretanje kompletne analize"""
    print("POKRETANJE NAPREDNE DATA SCIENCE ANALIZE")
    print("="*60)
    
    # Inicijalizuj
    analytics = AdvancedAnalytics()
    
    try:
        # Pokreni sve analize
        analytics.demographic_analysis()
        analytics.ai_tool_clustering() 
        analytics.quiz_performance_analysis()
        analytics.network_analysis()
        analytics.correlation_analysis()
        analytics.usage_pattern_analysis()
        
        # Generiši insights
        analytics.generate_insights()
        
        # Export za spoljne alate
        analytics.export_for_external_tools()
        
        print("\\n" + "="*60)
        print("ANALIZA ZAVRŠENA USPEŠNO!")
        print("Proverite 'data/exports/' direktorijum za fajlove za spoljne alate.")
        
    except Exception as e:
        print(f"Greška tokom analize: {e}")
        print("Ubedite se da postoje CSV fajlovi u 'data/' direktorijumu")

if __name__ == "__main__":
    main()
