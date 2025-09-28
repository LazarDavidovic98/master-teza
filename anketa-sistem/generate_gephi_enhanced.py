import pandas as pd
import csv
import os
from collections import defaultdict
import numpy as np

# Putanje do fajlova
data_dir = 'data'
csv_files = {
    'participants': os.path.join(data_dir, 'participants.csv'),
    'relationships': os.path.join(data_dir, 'relationships.csv'),
    'tool_usage': os.path.join(data_dir, 'tool_usage_patterns.csv'),
    'ai_knowledge': os.path.join(data_dir, 'ai_knowledge.csv')
}

# Čitanje podataka
participants_df = pd.read_csv(csv_files['participants'])
relationships_df = pd.read_csv(csv_files['relationships'])
tool_usage_df = pd.read_csv(csv_files['tool_usage'])

# Kreiranje čvorova (nodes) sa dodacima za Gephi
nodes = {}

# Definisanje boja za različite tipove čvorova
colors = {
    'participant': '#3498db',      # plava za učesnike
    'target_group': '#e74c3c',     # crvena za ciljane grupe
    'ai_tool': '#2ecc71',          # zelena za AI alate
    'organization': '#f39c12'      # narandžasta za organizacije
}

# Dodavanje učesnika kao čvorova
for _, participant in participants_df.iterrows():
    nodes[participant['participant_id']] = {
        'Id': participant['participant_id'],
        'Label': f"Učesnik {participant['participant_id'][:8]}",
        'Type': 'participant',
        'Ciljana_grupa': participant['ciljana_grupa'],
        'Age_group': participant['age_group'],
        'Country': participant['country'],
        'Education': participant['education'],
        'Work_experience': participant['work_experience'],
        'Institution': participant.get('institution', ''),
        'Size': 25,
        'Color': colors['participant'],
        'Shape': 'circle'
    }

# Dodavanje grupa kao čvorova sa detaljnijim informacijama
target_groups = participants_df['ciljana_grupa'].unique()
group_stats = {}

for group in target_groups:
    group_participants = participants_df[participants_df['ciljana_grupa'] == group]
    group_id = f"GROUP_{group.upper()}"
    group_count = len(group_participants)
    
    # Analiza karakteristika grupe
    avg_experience = group_participants['work_experience'].value_counts().index[0] if not group_participants.empty else ''
    most_common_education = group_participants['education'].value_counts().index[0] if not group_participants.empty else ''
    most_common_country = group_participants['country'].value_counts().index[0] if not group_participants.empty else ''
    
    nodes[group_id] = {
        'Id': group_id,
        'Label': f"Grupa: {group}",
        'Type': 'target_group',
        'Ciljana_grupa': group,
        'Age_group': '',
        'Country': most_common_country,
        'Education': most_common_education,
        'Work_experience': avg_experience,
        'Institution': f"{group_count} članova",
        'Size': group_count * 8,
        'Color': colors['target_group'],
        'Shape': 'diamond'
    }
    
    group_stats[group] = {
        'count': group_count,
        'education': most_common_education,
        'country': most_common_country
    }

# Dodavanje AI alata kao čvorova sa kategorijama
tool_categories = {
    'ChatGPT': 'conversational_ai',
    'Claude': 'conversational_ai', 
    'Gemini': 'conversational_ai',
    'GitHub_Copilot': 'code_assistant',
    'Grammarly': 'writing_assistant',
    'Midjourney': 'image_generation',
    'Stable_Diffusion': 'image_generation',
    'DALL-E': 'image_generation',
    'Bing_Chat': 'search_assistant',
    'DeepL': 'translation'
}

unique_tools = set()
for _, rel in relationships_df.iterrows():
    if rel['target_id'].startswith('TOOL_'):
        unique_tools.add(rel['target_id'])

for tool in unique_tools:
    tool_name = tool.replace('TOOL_', '')
    tool_usage_count = len(relationships_df[relationships_df['target_id'] == tool])
    tool_category = tool_categories.get(tool_name, 'other')
    
    # Analiza korišćenja po grupama
    tool_users_by_group = relationships_df[relationships_df['target_id'] == tool]['ciljana_grupa'].value_counts()
    most_used_by_group = tool_users_by_group.index[0] if not tool_users_by_group.empty else ''
    
    nodes[tool] = {
        'Id': tool,
        'Label': tool_name,
        'Type': 'ai_tool',
        'Ciljana_grupa': most_used_by_group,
        'Age_group': '',
        'Country': '',
        'Education': tool_category,
        'Work_experience': '',
        'Institution': f"Koristi {tool_usage_count} učesnika",
        'Size': min(tool_usage_count * 3, 50),  # Ograničavanje veličine
        'Color': colors['ai_tool'],
        'Shape': 'square'
    }

# Dodavanje organizacija kao čvorova
unique_orgs = set()
for _, rel in relationships_df.iterrows():
    if rel['target_id'].startswith('ORG_'):
        unique_orgs.add(rel['target_id'])

for org in unique_orgs:
    org_name = org.replace('ORG_', '')
    org_users = relationships_df[relationships_df['target_id'] == org]
    org_groups = org_users['ciljana_grupa'].value_counts()
    primary_group = org_groups.index[0] if not org_groups.empty else ''
    
    nodes[org] = {
        'Id': org,
        'Label': f"Org: {org_name}",
        'Type': 'organization',
        'Ciljana_grupa': primary_group,
        'Age_group': '',
        'Country': '',
        'Education': '',
        'Work_experience': '',
        'Institution': f"{len(org_users)} zaposlenih",
        'Size': 30,
        'Color': colors['organization'],
        'Shape': 'triangle'
    }

# Kreiranje poboljšanih veza (edges)
edges = []

# Dodavanje postojećih veza iz relationships.csv sa dodatnim atributima
for _, rel in relationships_df.iterrows():
    source_group = rel.get('ciljana_grupa', '')
    
    # Određivanje težine veze na osnovu tipa odnosa
    weight_mapping = {
        'uses_tool': 0.3,
        'belongs_to_group': 1.0,
        'affiliated_with': 0.8,
        'similar_usage': 0.5
    }
    
    edge_weight = weight_mapping.get(rel['relationship_type'], float(rel['weight']))
    
    edge = {
        'Source': rel['source_id'],
        'Target': rel['target_id'],
        'Type': 'Directed',
        'Weight': edge_weight,
        'Relationship_type': rel['relationship_type'],
        'Context': rel['context'],
        'Ciljana_grupa': source_group,
        'Label': f"{rel['relationship_type']}",
        'Color': '#34495e',  # Tamno siva za standardne veze
        'Thickness': max(1, int(edge_weight * 5))
    }
    edges.append(edge)

# Dodavanje veza učesnik-grupa
for group in target_groups:
    group_participants = participants_df[participants_df['ciljana_grupa'] == group]['participant_id'].tolist()
    
    for participant in group_participants:
        edge = {
            'Source': participant,
            'Target': f"GROUP_{group.upper()}",
            'Type': 'Undirected',
            'Weight': 1.0,
            'Relationship_type': 'member_of_group',
            'Context': 'group_membership',
            'Ciljana_grupa': group,
            'Label': 'pripada_grupi',
            'Color': colors['target_group'],
            'Thickness': 3
        }
        edges.append(edge)

# Dodavanje veza između grupa na osnovu sličnosti
group_tool_usage = defaultdict(set)
for _, rel in relationships_df.iterrows():
    if rel['target_id'].startswith('TOOL_') and rel['relationship_type'] == 'uses_tool':
        group_tool_usage[rel['ciljana_grupa']].add(rel['target_id'])

groups = list(group_tool_usage.keys())
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        group1, group2 = groups[i], groups[j]
        common_tools = group_tool_usage[group1] & group_tool_usage[group2]
        all_tools = group_tool_usage[group1] | group_tool_usage[group2]
        
        if len(common_tools) >= 2:
            similarity = len(common_tools) / len(all_tools) if all_tools else 0
            
            edge = {
                'Source': f"GROUP_{group1.upper()}",
                'Target': f"GROUP_{group2.upper()}",
                'Type': 'Undirected',
                'Weight': similarity,
                'Relationship_type': 'tool_similarity',
                'Context': 'inter_group_similarity',
                'Ciljana_grupa': f"{group1}_{group2}",
                'Label': f'slični_alati ({len(common_tools)})',
                'Color': '#9b59b6',  # Ljubičasta za sličnosti
                'Thickness': max(1, int(similarity * 8))
            }
            edges.append(edge)

# Kreiranje poboljšanih Gephi fajlova
nodes_df = pd.DataFrame(nodes.values())
nodes_df.to_csv('gephi_nodes_enhanced.csv', index=False, encoding='utf-8')

edges_df = pd.DataFrame(edges)
edges_df.to_csv('gephi_edges_enhanced.csv', index=False, encoding='utf-8')

# Kreiranje summary fajla
summary = {
    'total_nodes': len(nodes),
    'total_edges': len(edges),
    'node_types': nodes_df['Type'].value_counts().to_dict(),
    'edge_types': edges_df['Relationship_type'].value_counts().to_dict(),
    'group_statistics': group_stats
}

import json
with open('network_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"Generisani poboljšani Gephi fajlovi:")
print(f"- gephi_nodes_enhanced.csv: {len(nodes)} čvorova")
print(f"- gephi_edges_enhanced.csv: {len(edges)} veza")
print(f"- network_summary.json: statistike mreže")
print("\nTipovi čvorova sa bojama:")
for node_type in nodes_df['Type'].value_counts().items():
    color = colors.get(node_type[0], '#000000')
    print(f"  {node_type[0]}: {node_type[1]} čvorova (boja: {color})")

print("\nCiljane grupe kao centralni čvorovi:")
for group in target_groups:
    group_id = f"GROUP_{group.upper()}"
    if group_id in nodes:
        size = nodes[group_id]['Size']
        members = nodes[group_id]['Institution']
        print(f"  {group}: {members} (veličina čvora: {size})")

print("\nVeze između grupa:")
inter_group_edges = [e for e in edges if e['Relationship_type'] == 'tool_similarity']
for edge in inter_group_edges:
    print(f"  {edge['Source']} <-> {edge['Target']}: {edge['Label']}")
