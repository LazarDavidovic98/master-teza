import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

def generate_bipartite_gephi_files():
    """
    Generiše bipartitni (dvodelni) graf sa korisnicima i AI alatima
    Korisnike i alate stavlja u dve odvojene grupe
    Povezuje ih samo vezama "koristi alat"
    """
    print("🔄 GENERISANJE BIPARTITNOG (DVODELNOG) GRAFA")
    print("=" * 60)
    
    # Učitavanje podataka
    data_dir = 'data'
    
    print("📂 Učitavanje podataka...")
    try:
        participants_df = pd.read_csv(os.path.join(data_dir, 'participants.csv'))
        ai_knowledge_df = pd.read_csv(os.path.join(data_dir, 'ai_knowledge.csv'))
        relationships_df = pd.read_csv(os.path.join(data_dir, 'relationships.csv'))
        
        print(f"✅ Učitano {len(participants_df)} učesnika")
        print(f"✅ Učitano {len(ai_knowledge_df)} AI korišćenja")
        print(f"✅ Učitano {len(relationships_df)} veza")
    except Exception as e:
        print(f"❌ Greška pri učitavanju: {e}")
        return
    
    # --- KREIRANJE ČVOROVA ---
    print("🎯 Kreiranje čvorova...")
    
    nodes = []
    
    # 1. KORISNICI (User partition)
    print("👥 Dodavanje korisnika...")
    user_count = 0
    for _, participant in participants_df.iterrows():
        # Izračunaj broj alata koje korisnik koristi
        tools_used = len(ai_knowledge_df[ai_knowledge_df['participant_id'] == participant['participant_id']])
        
        # Određuj boju na osnovu ciljane grupe
        group_colors = {
            'prosveta': '#4ECDC4',
            'medicina': '#45B7D1', 
            'kreativna_filmska': '#FF6B6B',
            'it_industrija': '#96CEB4',
            'drustvene_nauke': '#FFEAA7',
            'ostalo': '#DDA0DD'
        }
        
        node = {
            'Id': participant['participant_id'],
            'Label': f"{participant['participant_id'][:8]}",
            'Type': 'user',
            'Partition': 'users',  # Bipartitni identifikator
            'CiljanaGrupa': participant['ciljana_grupa'],
            'Country': participant['country'],
            'Education': participant['education'],
            'WorkExperience': participant['work_experience'],
            'AgeGroup': participant['age_group'],
            'ExperienceLevel': participant['experience_level'],
            'Institution': participant.get('institution', ''),
            'ToolsUsed': tools_used,
            'Size': max(15, min(40, 15 + tools_used * 3)),  # Veličina prema broju alata
            'Color': group_colors.get(participant['ciljana_grupa'], '#808080'),
            'Shape': 'circle',
            'X': np.random.uniform(-150, -50, 1)[0],  # Leva strana
            'Y': np.random.uniform(-100, 100, 1)[0]
        }
        nodes.append(node)
        user_count += 1
    
    print(f"  ✅ Dodano {user_count} korisnika")
    
    # 2. AI ALATI (Tool partition)
    print("🤖 Dodavanje AI alata...")
    
    # Pronađi sve jedinstvene alate
    unique_tools = ai_knowledge_df['ai_tool'].unique()
    tool_count = 0
    
    for tool in unique_tools:
        # Izračunaj koliko korisnika koristi ovaj alat
        users_count = len(ai_knowledge_df[ai_knowledge_df['ai_tool'] == tool]['participant_id'].unique())
        
        # Kategorije alata
        tool_categories = {
            'ChatGPT': 'conversational_ai',
            'Claude': 'conversational_ai', 
            'Bing Chat': 'conversational_ai',
            'Gemini': 'conversational_ai',
            'GitHub Copilot': 'code_assistant',
            'DALL-E': 'image_generation',
            'Midjourney': 'image_generation',
            'Stable Diffusion': 'image_generation',
            'DeepL': 'translation',
            'Grammarly': 'writing_assistant'
        }
        
        # Boje prema kategoriji
        category_colors = {
            'conversational_ai': '#FF4757',     # Crvena
            'code_assistant': '#5352ED',        # Plava
            'image_generation': '#FF6348',      # Narandžasta  
            'translation': '#1DD1A1',           # Zelena
            'writing_assistant': '#FFC048',     # Žuta
            'other': '#747D8C'                  # Siva
        }
        
        tool_category = tool_categories.get(tool, 'other')
        
        node = {
            'Id': f"TOOL_{tool.replace(' ', '_')}",
            'Label': tool,
            'Type': 'ai_tool',
            'Partition': 'tools',  # Bipartitni identifikator
            'Category': tool_category,
            'UsersCount': users_count,
            'Size': max(20, min(50, 20 + users_count * 2)),  # Veličina prema broju korisnika
            'Color': category_colors.get(tool_category, '#747D8C'),
            'Shape': 'square',
            'X': np.random.uniform(50, 150, 1)[0],  # Desna strana
            'Y': np.random.uniform(-100, 100, 1)[0]
        }
        nodes.append(node)
        tool_count += 1
    
    print(f"  ✅ Dodano {tool_count} AI alata")
    
    # --- KREIRANJE VEZA ---
    print("🔗 Kreiranje veza (samo user-tool)...")
    
    edges = []
    edge_count = 0
    
    # Samo veze između korisnika i alata (bipartitne veze)
    for _, knowledge in ai_knowledge_df.iterrows():
        user_id = knowledge['participant_id']
        tool_id = f"TOOL_{knowledge['ai_tool'].replace(' ', '_')}"
        
        # Izračunaj težinu veze (može biti na osnovu frequency ili purpose)
        purposes = str(knowledge['purpose']).split(',') if pd.notna(knowledge['purpose']) else []
        weight = max(0.2, min(1.0, len(purposes) * 0.2))  # Više svrha = veća težina
        
        edge = {
            'Source': user_id,
            'Target': tool_id,
            'Type': 'Undirected',  # Bipartitni graf je obično undirected
            'Weight': weight,
            'Context': 'tool_usage',
            'Relationship_type': 'uses_tool',
            'Label': 'koristi',
            'Color': '#00CEC9',  # Teal boja za sve user-tool veze
            'Thickness': max(1, int(weight * 5)),
            'Purposes': str(knowledge['purpose']) if pd.notna(knowledge['purpose']) else ''
        }
        edges.append(edge)
        edge_count += 1
    
    print(f"  ✅ Dodano {edge_count} bipartitnih veza")
    
    # --- KREIRANJE DATAFRAME-OVA ---
    print("📊 Kreiranje DataFrame-ova...")
    
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    # --- SNIMANJE FAJLOVA ---
    print("💾 Snimanje bipartitnih Gephi fajlova...")
    
    nodes_filename = 'gephi_nodes_bipartite.csv'
    edges_filename = 'gephi_edges_bipartite.csv'
    
    nodes_df.to_csv(nodes_filename, index=False, encoding='utf-8')
    edges_df.to_csv(edges_filename, index=False, encoding='utf-8')
    
    # --- STATISTIKE I ANALIZA ---
    print("📈 Generisanje statistika...")
    
    # Analiza bipartitnog grafa
    user_nodes = nodes_df[nodes_df['Partition'] == 'users']
    tool_nodes = nodes_df[nodes_df['Partition'] == 'tools']
    
    # Analiza grupa korisnika
    group_stats = user_nodes['CiljanaGrupa'].value_counts().to_dict()
    
    # Analiza kategorija alata
    tool_category_stats = tool_nodes['Category'].value_counts().to_dict()
    
    # Najkorišćeniji alati
    tool_usage = tool_nodes.sort_values('UsersCount', ascending=False).head(5)
    top_tools = [(row['Label'], row['UsersCount']) for _, row in tool_usage.iterrows()]
    
    # Najaktivniji korisnici
    active_users = user_nodes.sort_values('ToolsUsed', ascending=False).head(5)
    top_users = [(row['Id'][:8], row['ToolsUsed'], row['CiljanaGrupa']) for _, row in active_users.iterrows()]
    
    # Kreiranje summary-ja
    summary = {
        'bipartite_graph_info': {
            'type': 'bipartite',
            'description': 'Dvodelni graf sa korisnicima i AI alatima',
            'partitions': ['users', 'tools']
        },
        'total_nodes': len(nodes_df),
        'users_count': len(user_nodes),
        'tools_count': len(tool_nodes),
        'total_edges': len(edges_df),
        'bipartite_density': len(edges_df) / (len(user_nodes) * len(tool_nodes)) if len(user_nodes) > 0 and len(tool_nodes) > 0 else 0,
        'user_groups': group_stats,
        'tool_categories': tool_category_stats,
        'top_5_tools': dict(top_tools),
        'top_5_active_users': [{'id': u[0], 'tools_used': u[1], 'group': u[2]} for u in top_users],
        'graph_properties': {
            'users_left_side': 'X coordinates: -150 to -50',
            'tools_right_side': 'X coordinates: 50 to 150',
            'edge_type': 'only user-tool connections',
            'no_user_user_edges': True,
            'no_tool_tool_edges': True
        },
        'suggested_analyses': [
            'Projection na korisnike: koji korisnici dele slične alate',
            'Projection na alate: koji alati su često korišćeni zajedno',
            'Community detection u projeksijama',
            'Centrality measures za identifikaciju ključnih čvorova',
            'Collaborative filtering za preporuke alata'
        ]
    }
    
    # Snimanje summary-ja
    summary_filename = 'network_summary_bipartite.json'
    with open(summary_filename, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # --- PRIKAZ REZULTATA ---
    print("\n" + "=" * 60)
    print("🎉 USPEŠNO GENERISAN BIPARTITNI (DVODELNI) GRAF!")
    print("=" * 60)
    
    print(f"📁 Kreirani fajlovi:")
    print(f"  📊 {nodes_filename}: {len(nodes_df)} čvorova")
    print(f"  🔗 {edges_filename}: {len(edges_df)} veza")
    print(f"  📄 {summary_filename}: statistike i analiza")
    
    print(f"\n🎯 Struktura bipartitnog grafa:")
    print(f"  👥 Korisnici (leva strana): {len(user_nodes)}")
    print(f"  🤖 AI alati (desna strana): {len(tool_nodes)}")
    print(f"  🔗 Bipartitne veze: {len(edges_df)}")
    print(f"  📊 Gustina grafa: {summary['bipartite_density']:.3f}")
    
    print(f"\n👥 Grupe korisnika:")
    for group, count in group_stats.items():
        print(f"  • {group}: {count} korisnika")
    
    print(f"\n🤖 Kategorije alata:")
    for category, count in tool_category_stats.items():
        print(f"  • {category}: {count} alata")
    
    print(f"\n🏆 Top 5 najkorišćenijih alata:")
    for i, (tool, users) in enumerate(top_tools, 1):
        print(f"  {i}. {tool}: {users} korisnika")
    
    print(f"\n🔥 Top 5 najaktivnijih korisnika:")
    for i, user_data in enumerate(top_users, 1):
        print(f"  {i}. {user_data[0]}: {user_data[1]} alata ({user_data[2]})")
    
    print(f"\n💡 Predložene analize:")
    for analysis in summary['suggested_analyses']:
        print(f"  • {analysis}")
    
    print(f"\n📋 Napomene za Gephi:")
    print(f"  🎨 Koristite 'Partition' atribut za bojenje čvorova")
    print(f"  📍 X koordinate su već podešene (korisnici levo, alati desno)")
    print(f"  🔍 Koristite Force Atlas 2 za finalno pozicioniranje")
    print(f"  📊 Size atribut odražava aktivnost čvorova")

if __name__ == "__main__":
    generate_bipartite_gephi_files()
