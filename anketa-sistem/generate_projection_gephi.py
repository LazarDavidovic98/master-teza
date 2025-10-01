import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict
from itertools import combinations

def generate_projection_gephi_files():
    """
    Generiše projekcije bipartitnog grafa:
    1. User-User projekcija (korisnici povezani preko zajedničkih alata)
    2. Tool-Tool projekcija (alati povezani preko zajedničkih korisnika)
    """
    print("🔄 GENERISANJE PROJEKCIJA BIPARTITNOG GRAFA")
    print("=" * 70)
    
    # Učitavanje podataka
    data_dir = 'data'
    
    print("📂 Učitavanje podataka...")
    try:
        participants_df = pd.read_csv(os.path.join(data_dir, 'participants.csv'))
        ai_knowledge_df = pd.read_csv(os.path.join(data_dir, 'ai_knowledge.csv'))
        
        print(f"✅ Učitano {len(participants_df)} učesnika")
        print(f"✅ Učitano {len(ai_knowledge_df)} AI korišćenja")
    except Exception as e:
        print(f"❌ Greška pri učitavanju: {e}")
        return
    
    # Kreiranje matrice user-tool
    print("🧮 Kreiranje user-tool matrice...")
    
    # Debug informacije
    print(f"  📋 Ukupno AI korišćenja: {len(ai_knowledge_df)}")
    print(f"  👤 Jedinstveni korisnici: {len(ai_knowledge_df['participant_id'].unique())}")
    print(f"  🤖 Jedinstveni alati: {len(ai_knowledge_df['ai_tool'].unique())}")
    
    # Prvo deduplikuj kombinacije participant_id i ai_tool
    ai_knowledge_unique = ai_knowledge_df[['participant_id', 'ai_tool']].drop_duplicates()
    print(f"  🔄 Jedinstvene kombinacije: {len(ai_knowledge_unique)}")
    
    # Kreiraj dummy vrednost za agregaciju
    ai_knowledge_unique['dummy'] = 1
    
    # Kreiraj pivot tabelu
    user_tool_matrix = ai_knowledge_unique.pivot_table(
        index='participant_id',
        columns='ai_tool',
        values='dummy',
        aggfunc='sum',
        fill_value=0
    )
    
    # Binarna matrica (1 ako koristi, 0 ako ne)
    user_tool_binary = (user_tool_matrix > 0).astype(int)
    
    print(f"  📊 Finalna matrica: {user_tool_binary.shape[0]} korisnika × {user_tool_binary.shape[1]} alata")
    
    print(f"  📊 Matrica dimenzije: {user_tool_binary.shape[0]} korisnika × {user_tool_binary.shape[1]} alata")
    
    # === USER-USER PROJEKCIJA ===
    print("\n👥 KREIRANJE USER-USER PROJEKCIJE")
    print("-" * 40)
    
    # Izračunaj sličnost između korisnika (Jaccard similarity)
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0
    
    # User nodes za projekciju
    user_nodes = []
    user_edges = []
    
    # Kreiranje user čvorova
    print("👤 Kreiranje user čvorova...")
    for user_id in user_tool_binary.index:
        user_info = participants_df[participants_df['participant_id'] == user_id]
        if user_info.empty:
            continue
        user_info = user_info.iloc[0]
        tools_used = set(user_tool_binary.columns[user_tool_binary.loc[user_id] == 1])
        
        group_colors = {
            'prosveta': '#4ECDC4',
            'medicina': '#45B7D1', 
            'kreativna_filmska': '#FF6B6B',
            'it_industrija': '#96CEB4',
            'drustvene_nauke': '#FFEAA7',
            'ostalo': '#DDA0DD'
        }
        
        node = {
            'Id': user_id,
            'Label': f"{user_id[:8]}",
            'Type': 'user',
            'CiljanaGrupa': user_info['ciljana_grupa'],
            'Country': user_info['country'],
            'Education': user_info['education'],
            'AgeGroup': user_info['age_group'],
            'ToolsCount': len(tools_used),
            'ToolsList': ', '.join(sorted(tools_used)),
            'Size': max(15, min(40, 15 + len(tools_used) * 2)),
            'Color': group_colors.get(user_info['ciljana_grupa'], '#808080'),
            'Shape': 'circle'
        }
        user_nodes.append(node)
    
    print(f"  ✅ Kreirano {len(user_nodes)} user čvorova")
    
    # Kreiranje user-user veza
    print("🔗 Kreiranje user-user veza...")
    users = list(user_tool_binary.index)
    edge_count = 0
    
    for user1, user2 in combinations(users, 2):
        tools1 = set(user_tool_binary.columns[user_tool_binary.loc[user1] == 1])
        tools2 = set(user_tool_binary.columns[user_tool_binary.loc[user2] == 1])
        
        similarity = jaccard_similarity(tools1, tools2)
        
        # Dodaj vezu samo ako je sličnost veća od praga
        if similarity > 0.15:  # Prag sličnosti
            common_tools = tools1.intersection(tools2)
            
            edge = {
                'Source': user1,
                'Target': user2,
                'Type': 'Undirected',
                'Weight': round(similarity, 3),
                'Context': 'shared_tools',
                'CommonTools': len(common_tools),
                'CommonToolsList': ', '.join(sorted(common_tools)),
                'Label': f'{len(common_tools)} alata',
                'Color': '#3498db',
                'Thickness': max(1, int(similarity * 10))
            }
            user_edges.append(edge)
            edge_count += 1
    
    print(f"  ✅ Kreirano {edge_count} user-user veza")
    
    # Snimanje user-user projekcije
    user_nodes_df = pd.DataFrame(user_nodes)
    user_edges_df = pd.DataFrame(user_edges)
    
    user_nodes_df.to_csv('gephi_nodes_user_projection.csv', index=False, encoding='utf-8')
    user_edges_df.to_csv('gephi_edges_user_projection.csv', index=False, encoding='utf-8')
    
    # === TOOL-TOOL PROJEKCIJA ===
    print("\n🤖 KREIRANJE TOOL-TOOL PROJEKCIJE")
    print("-" * 40)
    
    # Tool nodes za projekciju
    tool_nodes = []
    tool_edges = []
    
    # Kreiranje tool čvorova
    print("🔧 Kreiranje tool čvorova...")
    for tool in user_tool_binary.columns:
        users_using_tool = set(user_tool_binary.index[user_tool_binary[tool] == 1])
        
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
        
        category_colors = {
            'conversational_ai': '#FF4757',
            'code_assistant': '#5352ED',
            'image_generation': '#FF6348',
            'translation': '#1DD1A1',
            'writing_assistant': '#FFC048',
            'other': '#747D8C'
        }
        
        tool_category = tool_categories.get(tool, 'other')
        
        node = {
            'Id': f"TOOL_{tool.replace(' ', '_')}",
            'Label': tool,
            'Type': 'tool',
            'Category': tool_category,
            'UsersCount': len(users_using_tool),
            'UsersList': ', '.join(sorted([u[:8] for u in users_using_tool])),
            'Size': max(20, min(50, 20 + len(users_using_tool) * 2)),
            'Color': category_colors.get(tool_category, '#747D8C'),
            'Shape': 'square'
        }
        tool_nodes.append(node)
    
    print(f"  ✅ Kreirano {len(tool_nodes)} tool čvorova")
    
    # Kreiranje tool-tool veza
    print("🔗 Kreiranje tool-tool veza...")
    tools = list(user_tool_binary.columns)
    edge_count = 0
    
    for tool1, tool2 in combinations(tools, 2):
        users1 = set(user_tool_binary.index[user_tool_binary[tool1] == 1])
        users2 = set(user_tool_binary.index[user_tool_binary[tool2] == 1])
        
        similarity = jaccard_similarity(users1, users2)
        
        # Dodaj vezu samo ako je sličnost veća od praga
        if similarity > 0.1:  # Prag sličnosti za alate
            common_users = users1.intersection(users2)
            
            edge = {
                'Source': f"TOOL_{tool1.replace(' ', '_')}",
                'Target': f"TOOL_{tool2.replace(' ', '_')}",
                'Type': 'Undirected',
                'Weight': round(similarity, 3),
                'Context': 'shared_users',
                'CommonUsers': len(common_users),
                'CommonUsersList': ', '.join(sorted([u[:8] for u in common_users])),
                'Label': f'{len(common_users)} korisnika',
                'Color': '#e74c3c',
                'Thickness': max(1, int(similarity * 8))
            }
            tool_edges.append(edge)
            edge_count += 1
    
    print(f"  ✅ Kreirano {edge_count} tool-tool veza")
    
    # Snimavanje tool-tool projekcije
    tool_nodes_df = pd.DataFrame(tool_nodes)
    tool_edges_df = pd.DataFrame(tool_edges)
    
    tool_nodes_df.to_csv('gephi_nodes_tool_projection.csv', index=False, encoding='utf-8')
    tool_edges_df.to_csv('gephi_edges_tool_projection.csv', index=False, encoding='utf-8')
    
    # === STATISTIKE I ANALIZA ===
    print("\n📊 GENERISANJE STATISTIKA...")
    
    # User projection analiza
    user_degrees = pd.Series(dtype=float)
    if len(user_edges_df) > 0:
        user_degrees = user_edges_df['Source'].value_counts().add(
            user_edges_df['Target'].value_counts(), fill_value=0
        ).sort_values(ascending=False)
    
    # Tool projection analiza
    tool_degrees = pd.Series(dtype=float)
    if len(tool_edges_df) > 0:
        tool_degrees = tool_edges_df['Source'].value_counts().add(
            tool_edges_df['Target'].value_counts(), fill_value=0
        ).sort_values(ascending=False)
    
    # Kreiranje summary-ja
    projections_summary = {
        'projection_info': {
            'type': 'bipartite_projections',
            'description': 'Projekcije bipartitnog grafa na pojedinačne skupove',
            'projections': ['user-user', 'tool-tool']
        },
        'user_projection': {
            'nodes': len(user_nodes_df),
            'edges': len(user_edges_df),
            'density': (2 * len(user_edges_df)) / (len(user_nodes_df) * (len(user_nodes_df) - 1)) if len(user_nodes_df) > 1 else 0,
            'avg_similarity': user_edges_df['Weight'].mean() if len(user_edges_df) > 0 else 0,
            'top_connected_users': user_degrees.head(5).to_dict() if len(user_degrees) > 0 else {}
        },
        'tool_projection': {
            'nodes': len(tool_nodes_df),
            'edges': len(tool_edges_df),
            'density': (2 * len(tool_edges_df)) / (len(tool_nodes_df) * (len(tool_nodes_df) - 1)) if len(tool_nodes_df) > 1 else 0,
            'avg_similarity': tool_edges_df['Weight'].mean() if len(tool_edges_df) > 0 else 0,
            'top_connected_tools': tool_degrees.head(5).to_dict() if len(tool_degrees) > 0 else {}
        },
        'analysis_possibilities': {
            'user_projection': [
                'Community detection - grupe korisnika sa sličnim preferencama',
                'Centrality measures - identifikacija influencer korisnika',
                'Clustering coefficient - koliko su korisnici klasterirani',
                'Collaborative filtering - preporučivanje novih alata'
            ],
            'tool_projection': [
                'Tool clustering - grupiranje sličnih alata',
                'Market basket analysis - alati koji se često koriste zajedno',
                'Tool recommendation systems - predlog kombinacija alata',
                'Technology stack analysis - identifikacija tech stackova'
            ]
        }
    }
    
    # Snimavanje summary-ja
    with open('network_summary_projections.json', 'w', encoding='utf-8') as f:
        json.dump(projections_summary, f, ensure_ascii=False, indent=2)
    
    # === PRIKAZ REZULTATA ===
    print("\n" + "=" * 70)
    print("🎉 USPEŠNO GENERISANE PROJEKCIJE BIPARTITNOG GRAFA!")
    print("=" * 70)
    
    print(f"📁 Kreirani fajlovi:")
    print(f"\n👥 USER-USER PROJEKCIJA:")
    print(f"  📊 gephi_nodes_user_projection.csv: {len(user_nodes_df)} čvorova")
    print(f"  🔗 gephi_edges_user_projection.csv: {len(user_edges_df)} veza")
    print(f"  📈 Gustina: {projections_summary['user_projection']['density']:.3f}")
    print(f"  🎯 Prosečna sličnost: {projections_summary['user_projection']['avg_similarity']:.3f}")
    
    print(f"\n🤖 TOOL-TOOL PROJEKCIJA:")
    print(f"  📊 gephi_nodes_tool_projection.csv: {len(tool_nodes_df)} čvorova")
    print(f"  🔗 gephi_edges_tool_projection.csv: {len(tool_edges_df)} veza")
    print(f"  📈 Gustina: {projections_summary['tool_projection']['density']:.3f}")
    print(f"  🎯 Prosečna sličnost: {projections_summary['tool_projection']['avg_similarity']:.3f}")
    
    print(f"\n📄 network_summary_projections.json: detaljne statistike")
    
    if len(user_degrees) > 0:
        print(f"\n🏆 Najkonektovniji korisnici (user projekcija):")
        for user, connections in user_degrees.head(3).items():
            user_short = user[:8]
            print(f"  • {user_short}: {int(connections)} veza")
    
    if len(tool_degrees) > 0:
        print(f"\n🔧 Najkonektovniji alati (tool projekcija):")
        for tool, connections in tool_degrees.head(3).items():
            tool_name = tool.replace('TOOL_', '').replace('_', ' ')
            print(f"  • {tool_name}: {int(connections)} veza")
    
    print(f"\n💡 Mogući tipovi analiza:")
    print(f"👥 User projekcija:")
    for analysis in projections_summary['analysis_possibilities']['user_projection']:
        print(f"  • {analysis}")
    
    print(f"\n🤖 Tool projekcija:")
    for analysis in projections_summary['analysis_possibilities']['tool_projection']:
        print(f"  • {analysis}")
    
    print(f"\n📋 Napomene za Gephi:")
    print(f"  🎨 Koristite 'CiljanaGrupa' (users) ili 'Category' (tools) za bojenje")
    print(f"  📏 'Weight' atribut predstavlja Jaccard sličnost")
    print(f"  🔍 Koristite Modularity za community detection")
    print(f"  📊 'Size' atribut odražava aktivnost/popularnost")

if __name__ == "__main__":
    generate_projection_gephi_files()
