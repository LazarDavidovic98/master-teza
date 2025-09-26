#!/usr/bin/env python3
"""
Process Real Survey Data
Obrađuje postojeći survey_responses.csv fajl i generiše data science dataset-ove
"""

import os
import sys
from data_science_extension import DataScienceManager

def process_real_data():
    """Obradi postojeće podatke iz survey_responses.csv"""
    
    print("🔬 PROCESSING REAL SURVEY DATA")
    print("=" * 50)
    
    # Kreiraj Data Science Manager
    data_manager = DataScienceManager()
    
    # Obradi postojeće podatke
    survey_file = "survey_responses.csv"
    
    if not os.path.exists(survey_file):
        print(f"❌ Fajl {survey_file} ne postoji!")
        print("💡 Pokrenite anketu prvo da generirate podatke")
        return
    
    print(f"📂 Čitam podatke iz {survey_file}...")
    processed_count = data_manager.process_existing_survey_data(survey_file)
    
    if processed_count == 0:
        print("❌ Nema podataka za obradu!")
        return
    
    # Kreiraj network graf
    print("\n🕸️ Kreiram network graf...")
    G = data_manager.create_network_graph()
    
    print(f"Graf kreiran sa {G.number_of_nodes()} čvorova i {G.number_of_edges()} veza")
    
    # Generiši Gephi fajlove
    print("\n📁 Generiram Gephi export fajlove...")
    nodes_file, edges_file = data_manager.generate_gephi_files()
    print(f"✅ Kreirani fajlovi:")
    print(f"   • {nodes_file}")
    print(f"   • {edges_file}")
    
    # Pokreni advanced analytics ako je moguće
    try:
        from advanced_analytics import AdvancedAnalytics
        print("\n📊 Pokretam advanced analytics...")
        
        analytics = AdvancedAnalytics()
        insights = analytics.generate_insights()
        
        print("\n" + "=" * 60)
        print("KEY INSIGHTS FROM REAL DATA")
        print("=" * 60)
        print(insights)
        
    except ImportError as e:
        print(f"\n⚠️ Advanced analytics nedostupne: {e}")
        print("💡 Instalirajte: pip install scikit-learn matplotlib seaborn")
    
    print("\n🎯 SLEDEĆI KORACI:")
    print("1. Pokrenite Flask app: python app.py")
    print("2. Idite na http://127.0.0.1:5000/data_science")
    print("3. Otvorite Gephi i importujte:")
    print("   • data/exports/gephi_edges.csv (kao Edges)")
    print("   • data/exports/gephi_nodes.csv (kao Nodes)")

def main():
    try:
        process_real_data()
    except Exception as e:
        print(f"❌ Greška: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
