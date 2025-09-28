import pandas as pd
import numpy as np
from collections import defaultdict
import os

def create_group_focused_network():
    """
    Kreira Gephi mrežu fokusiranu na ciljane grupe kao centralne čvorove
    """
    
    # Učitavanje podataka
    data_dir = 'data'
    participants_df = pd.read_csv(os.path.join(data_dir, 'participants.csv'))
    relationships_df = pd.read_csv(os.path.join(data_dir, 'relationships.csv'))
    tool_usage_df = pd.read_csv(os.path.join(data_dir, 'tool_usage_patterns.csv'))
    
    nodes = {}
    edges = []
    
    # 1. KREIRANJE ČVOROVA ZA CILJANE GRUPE (centralni čvorovi)
    target_groups = participants_df['ciljana_grupa'].unique()
    
    print("=== KREIRANJE MREŽE FOKUSIRANE NA CILJANE GRUPE ===\n")
    
    for group in target_groups:
        group_participants = participants_df[participants_df['ciljana_grupa'] == group]
        group_id = f"GROUP_{group.upper()}"
        
        # Analiza grupe
        member_count = len(group_participants)
        countries = group_participants['country'].value_counts()
        education_levels = group_participants['education'].value_counts()
        experience_levels = group_participants['work_experience'].value_counts()
        age_groups = group_participants['age_group'].value_counts()
        
        # Kreiranje opisnog labela
        top_country = countries.index[0] if len(countries) > 0 else 'N/A'
        top_education = education_levels.index[0] if len(education_levels) > 0 else 'N/A'
        top_experience = experience_levels.index[0] if len(experience_levels) > 0 else 'N/A'
        
        nodes[group_id] = {
            'Id': group_id,
            'Label': f"{group.upper()}\\n({member_count} članova)",
            'Type': 'target_group',
            'Group_Name': group,
            'Member_Count': member_count,
            'Top_Country': top_country,
            'Top_Education': top_education,
            'Top_Experience': top_experience,
            'Diversity_Score': len(countries) / member_count if member_count > 0 else 0,
            'Size': max(30, member_count * 5),
            'Color': '#e74c3c',  # Crvena za grupe
            'Shape': 'diamond',
            'X': 0,  # Centralne pozicije
            'Y': 0
        }
        
        print(f"Grupa: {group}")
        print(f"  - Članova: {member_count}")
        print(f"  - Najčešća zemlja: {top_country}")
        print(f"  - Najčešće obrazovanje: {top_education}")
        print(f"  - Najčešće iskustvo: {top_experience}")
        print()
    
    # 2. KREIRANJE ČVOROVA ZA UČESNIKE (povezani sa grupama)
    for _, participant in participants_df.iterrows():
        p_id = participant['participant_id']
        group = participant['ciljana_grupa']
        
        nodes[p_id] = {
            'Id': p_id,
            'Label': f"U_{p_id[:6]}",
            'Type': 'participant',
            'Group_Name': group,
            'Member_Count': 1,
            'Top_Country': participant['country'],
            'Top_Education': participant['education'],
            'Top_Experience': participant['work_experience'],
            'Diversity_Score': 0,
            'Size': 15,
            'Color': '#3498db',  # Plava za učesnike
            'Shape': 'circle',
            'X': np.random.uniform(-100, 100),
            'Y': np.random.uniform(-100, 100)
        }
        
        # Veza učesnik -> grupa (OBAVEZNA)
        edges.append({
            'Source': p_id,
            'Target': f"GROUP_{group.upper()}",
            'Type': 'Directed',
            'Weight': 1.0,
            'Relationship': 'belongs_to',
            'Label': 'pripada',
            'Color': '#34495e',
            'Thickness': 3
        })
    
    # 3. KREIRANJE ČVOROVA ZA AI ALATE (povezani sa grupama)
    tool_usage_by_group = defaultdict(lambda: defaultdict(int))
    
    # Analiza korišćenja alata po grupama
    for _, rel in relationships_df.iterrows():
        if rel['target_id'].startswith('TOOL_') and rel['relationship_type'] == 'uses_tool':
            group = rel['ciljana_grupa']
            tool = rel['target_id']
            tool_usage_by_group[group][tool] += 1
    
    # Kreiranje čvorova za alate sa informacijama o korišćenju
    all_tools = set()
    for _, rel in relationships_df.iterrows():
        if rel['target_id'].startswith('TOOL_'):
            all_tools.add(rel['target_id'])
    
    for tool in all_tools:
        tool_name = tool.replace('TOOL_', '')
        total_users = len(relationships_df[relationships_df['target_id'] == tool])
        
        # Nalaženje grupe koja najviše koristi ovaj alat
        max_usage = 0
        primary_group = ''
        for group, tools in tool_usage_by_group.items():
            if tools[tool] > max_usage:
                max_usage = tools[tool]
                primary_group = group
        
        nodes[tool] = {
            'Id': tool,
            'Label': tool_name,
            'Type': 'ai_tool',
            'Group_Name': primary_group,
            'Member_Count': total_users,
            'Top_Country': '',
            'Top_Education': '',
            'Top_Experience': '',
            'Diversity_Score': len([g for g, tools in tool_usage_by_group.items() if tools[tool] > 0]) / len(target_groups),
            'Size': min(25, total_users * 2),
            'Color': '#2ecc71',  # Zelena za alate
            'Shape': 'square',
            'X': np.random.uniform(-150, 150),
            'Y': np.random.uniform(-150, 150)
        }
        
        # Veze alat -> grupa (na osnovu korišćenja)
        for group, tools in tool_usage_by_group.items():
            if tools[tool] > 0:
                usage_intensity = tools[tool] / max(1, sum(tools.values()))
                
                edges.append({
                    'Source': tool,
                    'Target': f"GROUP_{group.upper()}",
                    'Type': 'Directed',
                    'Weight': usage_intensity,
                    'Relationship': 'used_by_group',
                    'Label': f'koristi_se ({tools[tool]}x)',
                    'Color': '#27ae60',
                    'Thickness': max(1, int(usage_intensity * 8))
                })
    
    # 4. VEZE IZMEĐU GRUPA (na osnovu sličnosti)
    print("=== ANALIZA SLIČNOSTI IZMEĐU GRUPA ===\n")
    
    groups_list = list(target_groups)
    for i in range(len(groups_list)):
        for j in range(i+1, len(groups_list)):
            group1, group2 = groups_list[i], groups_list[j]
            
            # Sličnost na osnovu alata
            tools1 = set(tool_usage_by_group[group1].keys())
            tools2 = set(tool_usage_by_group[group2].keys())
            common_tools = tools1 & tools2
            all_tools_both = tools1 | tools2
            
            tool_similarity = len(common_tools) / len(all_tools_both) if all_tools_both else 0
            
            # Sličnost na osnovu demografije
            demo1 = participants_df[participants_df['ciljana_grupa'] == group1]
            demo2 = participants_df[participants_df['ciljana_grupa'] == group2]
            
            common_countries = set(demo1['country']) & set(demo2['country'])
            common_education = set(demo1['education']) & set(demo2['education'])
            
            demo_similarity = (len(common_countries) + len(common_education)) / 10  # Normalizovano
            
            # Kombinovana sličnost
            overall_similarity = (tool_similarity + demo_similarity) / 2
            
            print(f"{group1} <-> {group2}:")
            print(f"  - Zajednički alati: {len(common_tools)} ({tool_similarity:.2f})")
            print(f"  - Demografska sličnost: {demo_similarity:.2f}")
            print(f"  - Ukupna sličnost: {overall_similarity:.2f}")
            print()
            
            if overall_similarity > 0.1:  # Prag za kreiranje veze
                edges.append({
                    'Source': f"GROUP_{group1.upper()}",
                    'Target': f"GROUP_{group2.upper()}",
                    'Type': 'Undirected',
                    'Weight': overall_similarity,
                    'Relationship': 'similar_groups',
                    'Label': f'sličnost ({overall_similarity:.2f})',
                    'Color': '#9b59b6',  # Ljubičasta za sličnost
                    'Thickness': max(1, int(overall_similarity * 10))
                })
    
    # 5. KREIRANJE GEPHI FAJLOVA
    nodes_df = pd.DataFrame(nodes.values())
    edges_df = pd.DataFrame(edges)
    
    # Sortiranje - grupe prvo, zatim učesnici, pa alati
    type_order = {'target_group': 0, 'participant': 1, 'ai_tool': 2}
    nodes_df['sort_order'] = nodes_df['Type'].map(type_order)
    nodes_df = nodes_df.sort_values('sort_order').drop('sort_order', axis=1)
    
    # Snimanje fajlova
    nodes_df.to_csv('gephi_nodes_group_focused.csv', index=False, encoding='utf-8')
    edges_df.to_csv('gephi_edges_group_focused.csv', index=False, encoding='utf-8')
    
    # Statistike
    print("=== FINALNE STATISTIKE ===")
    print(f"Ukupno čvorova: {len(nodes)}")
    print(f"  - Ciljane grupe: {len([n for n in nodes.values() if n['Type'] == 'target_group'])}")
    print(f"  - Učesnici: {len([n for n in nodes.values() if n['Type'] == 'participant'])}")
    print(f"  - AI alati: {len([n for n in nodes.values() if n['Type'] == 'ai_tool'])}")
    print(f"Ukupno veza: {len(edges)}")
    print(f"  - Pripadnost grupama: {len([e for e in edges if e['Relationship'] == 'belongs_to'])}")
    print(f"  - Korišćenje alata: {len([e for e in edges if e['Relationship'] == 'used_by_group'])}")
    print(f"  - Sličnost grupa: {len([e for e in edges if e['Relationship'] == 'similar_groups'])}")
    
    return nodes_df, edges_df

if __name__ == "__main__":
    nodes_df, edges_df = create_group_focused_network()
    print("\n✅ Uspešno generisani fajlovi:")
    print("   - gephi_nodes_group_focused.csv")  
    print("   - gephi_edges_group_focused.csv")
    print("\nOvi fajlovi su optimizovani za Gephi analizu sa ciljanim grupama kao centralnim čvorovima!")
