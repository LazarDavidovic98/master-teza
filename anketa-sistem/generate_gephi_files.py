import pandas as pd
import csv
import os
from collections import defaultdict

# Putanje do fajlova
data_dir = 'data'
csv_files = {
    'participants': os.path.join(data_dir, 'participants.csv'),
    'relationships': os.path.join(data_dir, 'relationships.csv'),
    'tool_usage': os.path.join(data_dir, 'tool_usage_patterns.csv'),
    'ai_knowledge': os.path.join(data_dir, 'ai_knowledge.csv'),
    'quiz_responses': os.path.join(data_dir, 'quiz_responses.csv')
}

# Čitanje podataka
participants_df = pd.read_csv(csv_files['participants'])
relationships_df = pd.read_csv(csv_files['relationships'])
tool_usage_df = pd.read_csv(csv_files['tool_usage'])
ai_knowledge_df = pd.read_csv(csv_files['ai_knowledge'])

# Kreiranje čvorova (nodes)
nodes = {}

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
        'Size': 20
    }

# Dodavanje grupa kao čvorova
target_groups = participants_df['ciljana_grupa'].unique()
for group in target_groups:
    group_id = f"GROUP_{group.upper()}"
    group_count = len(participants_df[participants_df['ciljana_grupa'] == group])
    
    nodes[group_id] = {
        'Id': group_id,
        'Label': f"Grupa: {group}",
        'Type': 'target_group',
        'Ciljana_grupa': group,
        'Age_group': '',
        'Country': '',
        'Education': '',
        'Work_experience': '',
        'Institution': '',
        'Size': group_count * 10  # Veličina proporcionalna broju članova
    }

# Dodavanje AI alata kao čvorova
unique_tools = set()
for _, rel in relationships_df.iterrows():
    if rel['target_id'].startswith('TOOL_'):
        unique_tools.add(rel['target_id'])

for tool in unique_tools:
    tool_name = tool.replace('TOOL_', '')
    tool_usage_count = len(relationships_df[relationships_df['target_id'] == tool])
    
    nodes[tool] = {
        'Id': tool,
        'Label': tool_name,
        'Type': 'ai_tool',
        'Ciljana_grupa': '',
        'Age_group': '',
        'Country': '',
        'Education': '',
        'Work_experience': '',
        'Institution': '',
        'Size': tool_usage_count * 2
    }

# Dodavanje organizacija kao čvorova
unique_orgs = set()
for _, rel in relationships_df.iterrows():
    if rel['target_id'].startswith('ORG_'):
        unique_orgs.add(rel['target_id'])

for org in unique_orgs:
    org_name = org.replace('ORG_', '')
    
    nodes[org] = {
        'Id': org,
        'Label': f"Org: {org_name}",
        'Type': 'organization',
        'Ciljana_grupa': '',
        'Age_group': '',
        'Country': '',
        'Education': '',
        'Work_experience': '',
        'Institution': '',
        'Size': 15
    }

# Kreiranje veza (edges)
edges = []

# Dodavanje postojećih veza iz relationships.csv
for _, rel in relationships_df.iterrows():
    source_group = rel.get('ciljana_grupa', '')
    
    edge = {
        'Source': rel['source_id'],
        'Target': rel['target_id'],
        'Type': 'Directed',
        'Weight': float(rel['weight']),
        'Relationship_type': rel['relationship_type'],
        'Context': rel['context'],
        'Ciljana_grupa': source_group,
        'Label': f"{rel['relationship_type']}"
    }
    edges.append(edge)

# Dodavanje veza između učesnika na osnovu zajedničkih karakteristika
# Veze na osnovu ciljanih grupa
for group in target_groups:
    group_participants = participants_df[participants_df['ciljana_grupa'] == group]['participant_id'].tolist()
    
    # Povezivanje učesnika sa njihovom grupom
    for participant in group_participants:
        edge = {
            'Source': participant,
            'Target': f"GROUP_{group.upper()}",
            'Type': 'Undirected',
            'Weight': 1.0,
            'Relationship_type': 'member_of_group',
            'Context': 'group_membership',
            'Ciljana_grupa': group,
            'Label': 'pripada_grupi'
        }
        edges.append(edge)

# Dodavanje veza između grupa na osnovu sličnosti u korišćenju alata
group_tool_usage = defaultdict(set)
for _, rel in relationships_df.iterrows():
    if rel['target_id'].startswith('TOOL_') and rel['relationship_type'] == 'uses_tool':
        group_tool_usage[rel['ciljana_grupa']].add(rel['target_id'])

# Kreiranje veza između grupa na osnovu zajedničkih alata
groups = list(group_tool_usage.keys())
for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        group1, group2 = groups[i], groups[j]
        common_tools = group_tool_usage[group1] & group_tool_usage[group2]
        
        if len(common_tools) >= 2:  # Ako dele barem 2 alata
            similarity = len(common_tools) / len(group_tool_usage[group1] | group_tool_usage[group2])
            
            edge = {
                'Source': f"GROUP_{group1.upper()}",
                'Target': f"GROUP_{group2.upper()}",
                'Type': 'Undirected',
                'Weight': similarity,
                'Relationship_type': 'tool_similarity',
                'Context': 'inter_group_similarity',
                'Ciljana_grupa': f"{group1}_{group2}",
                'Label': f'slični_alati ({len(common_tools)})'
            }
            edges.append(edge)

# Kreiranje Gephi nodes CSV fajla
nodes_df = pd.DataFrame(nodes.values())
nodes_df.to_csv('gephi_nodes.csv', index=False, encoding='utf-8')

# Kreiranje Gephi edges CSV fajla
edges_df = pd.DataFrame(edges)
edges_df.to_csv('gephi_edges.csv', index=False, encoding='utf-8')

print(f"Generisani fajlovi:")
print(f"- gephi_nodes.csv: {len(nodes)} čvorova")
print(f"- gephi_edges.csv: {len(edges)} veza")
print("\nTipovi čvorova:")
for node_type in nodes_df['Type'].value_counts().items():
    print(f"  {node_type[0]}: {node_type[1]} čvorova")

print("\nTipovi veza:")
for edge_type in edges_df['Relationship_type'].value_counts().items():
    print(f"  {edge_type[0]}: {edge_type[1]} veza")

print("\nCiljane grupe kao čvorovi:")
for group in target_groups:
    group_id = f"GROUP_{group.upper()}"
    group_size = nodes[group_id]['Size']
    print(f"  {group}: {group_size//10} članova")
