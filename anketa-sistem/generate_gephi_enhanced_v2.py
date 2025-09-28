import pandas as pd
import numpy as np
import json
import os

def enhance_gephi_files():
    """
    Unapređuje postojeće gephi_nodes.csv i gephi_edges.csv fajlove iz data/ foldera
    """
    print("🔄 UNAPREĐIVANJE GEPHI FAJLOVA")
    print("=" * 50)
    
    # Učitavanje postojećih fajlova
    data_dir = 'data'
    
    print("📂 Učitavanje postojećih fajlova...")
    try:
        nodes_df = pd.read_csv(os.path.join(data_dir, 'gephi_nodes.csv'))
        edges_df = pd.read_csv(os.path.join(data_dir, 'gephi_edges.csv'))
        participants_df = pd.read_csv(os.path.join(data_dir, 'participants.csv'))
        relationships_df = pd.read_csv(os.path.join(data_dir, 'relationships.csv'))
        
        print(f"✅ Učitano {len(nodes_df)} čvorova i {len(edges_df)} veza")
    except Exception as e:
        print(f"❌ Greška pri učitavanju: {e}")
        return
    
    # Unaprediti čvorove
    print("🎨 Unapređivanje čvorova...")
    
    # Dodaj Shape i koordinate
    def get_shape(node_type):
        shapes = {
            'participant': 'circle',
            'target_group': 'diamond',
            'ai_tool': 'square',
            'organization': 'triangle',
            'entity': 'triangle'
        }
        return shapes.get(node_type, 'circle')
    
    # Dodaj nove kolone
    nodes_df['Shape'] = nodes_df['Type'].apply(get_shape)
    nodes_df['X'] = np.random.uniform(-100, 100, len(nodes_df))
    nodes_df['Y'] = np.random.uniform(-100, 100, len(nodes_df))
    
    # Dodaj demografske informacije za učesnike
    for idx, row in nodes_df.iterrows():
        if row['Type'] == 'participant':
            participant_info = participants_df[participants_df['participant_id'] == row['Id']]
            if not participant_info.empty:
                p_info = participant_info.iloc[0]
                nodes_df.loc[idx, 'Country'] = p_info.get('country', '')
                nodes_df.loc[idx, 'Education'] = p_info.get('education', '')
                nodes_df.loc[idx, 'Work_experience'] = p_info.get('work_experience', '')
                nodes_df.loc[idx, 'Institution'] = p_info.get('organization', '')
    
    # Unaprediti veze
    print("🔗 Unapređivanje veza...")
    
    # Dodaj Type kolonu (Directed/Undirected)
    def get_edge_type(context):
        if context in ['target_group', 'ai_tool_usage', 'organizational']:
            return 'Directed'
        return 'Undirected'
    
    edges_df['Type'] = edges_df['Context'].apply(get_edge_type)
    
    # Dodaj Relationship_type kolonu
    def get_relationship_type(context):
        mapping = {
            'target_group': 'belongs_to_group',
            'ai_tool_usage': 'uses_tool',
            'organizational': 'affiliated_with'
        }
        return mapping.get(context, context)
    
    edges_df['Relationship_type'] = edges_df['Context'].apply(get_relationship_type)
    
    # Dodaj Label kolonu
    def get_label(relationship_type):
        mapping = {
            'belongs_to_group': 'pripada',
            'uses_tool': 'koristi',
            'affiliated_with': 'radi_u'
        }
        return mapping.get(relationship_type, 'povezano')
    
    edges_df['Label'] = edges_df['Relationship_type'].apply(get_label)
    
    # Dodaj Ciljana_grupa kolonu
    edges_df['Ciljana_grupa'] = ''
    for idx, row in edges_df.iterrows():
        rel_info = relationships_df[
            (relationships_df['source_id'] == row['Source']) & 
            (relationships_df['target_id'] == row['Target'])
        ]
        if not rel_info.empty:
            edges_df.loc[idx, 'Ciljana_grupa'] = rel_info.iloc[0].get('ciljana_grupa', '')
    
    # Dodaj veze između grupa na osnovu sličnosti
    print("🔄 Dodavanje veza sličnosti između grupa...")
    
    target_groups = nodes_df[nodes_df['Type'] == 'target_group']['Id'].tolist()
    similarity_edges = []
    
    for i in range(len(target_groups)):
        for j in range(i+1, len(target_groups)):
            group1_id = target_groups[i]
            group2_id = target_groups[j]
            
            # Pronađi alate koje koriste ove grupe
            group1_tools = set(edges_df[
                (edges_df['Target'] == group1_id) & 
                (edges_df['Source'].str.startswith('TOOL_'))
            ]['Source'].tolist())
            
            group2_tools = set(edges_df[
                (edges_df['Target'] == group2_id) & 
                (edges_df['Source'].str.startswith('TOOL_'))
            ]['Source'].tolist())
            
            if group1_tools and group2_tools:
                common_tools = group1_tools & group2_tools
                all_tools = group1_tools | group2_tools
                similarity = len(common_tools) / len(all_tools) if all_tools else 0
                
                if similarity > 0.1:  # Prag sličnosti
                    similarity_edge = {
                        'Source': group1_id,
                        'Target': group2_id,
                        'Type': 'Undirected',
                        'Weight': round(similarity, 2),
                        'Context': 'group_similarity',
                        'Color': '#9b59b6',
                        'Thickness': max(1, int(similarity * 8)),
                        'Relationship_type': 'group_similarity',
                        'Label': f'sličnost ({len(common_tools)})',
                        'Ciljana_grupa': f'{group1_id}_{group2_id}'
                    }
                    similarity_edges.append(similarity_edge)
    
    # Dodaj similarity edges u DataFrame
    if similarity_edges:
        similarity_df = pd.DataFrame(similarity_edges)
        edges_df = pd.concat([edges_df, similarity_df], ignore_index=True)
        print(f"✅ Dodano {len(similarity_edges)} veza sličnosti između grupa")
    
    # Snimi unapređene fajlove
    print("💾 Snimanje unapređenih fajlova...")
    
    nodes_df.to_csv('gephi_nodes_enhanced.csv', index=False, encoding='utf-8')
    edges_df.to_csv('gephi_edges_enhanced.csv', index=False, encoding='utf-8')
    
    # Kreiraj summary
    summary = {
        'total_nodes': len(nodes_df),
        'total_edges': len(edges_df),
        'node_types': nodes_df['Type'].value_counts().to_dict(),
        'edge_types': edges_df['Relationship_type'].value_counts().to_dict(),
        'enhanced_from': 'data/gephi_nodes.csv and data/gephi_edges.csv',
        'enhancements': [
            'Added Shape attribute for different node shapes',
            'Added X,Y coordinates for initial layout',
            'Enhanced demographic information for participants',
            'Added group similarity edges',
            'Improved edge labeling and typing',
            'Added explicit Directed/Undirected edge types'
        ]
    }
    
    with open('network_summary_enhanced.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Prikaz rezultata
    print("\n" + "=" * 50)
    print("🎉 USPEŠNO GENERISANI POBOLJŠANI GEPHI FAJLOVI!")
    print("=" * 50)
    print(f"📊 gephi_nodes_enhanced.csv: {len(nodes_df)} čvorova")
    print(f"🔗 gephi_edges_enhanced.csv: {len(edges_df)} veza")
    print(f"📄 network_summary_enhanced.json: statistike mreže")
    
    print(f"\n📈 Tipovi čvorova:")
    for node_type, count in nodes_df['Type'].value_counts().items():
        print(f"  • {node_type}: {count} čvorova")
    
    print(f"\n🔗 Tipovi veza:")
    for edge_type, count in edges_df['Relationship_type'].value_counts().items():
        print(f"  • {edge_type}: {count} veza")
    
    print(f"\n✨ Dodana poboljšanja:")
    print(f"  🎨 Shape atribut za različite oblike čvorova")
    print(f"  📍 X,Y koordinate za početni layout")
    print(f"  👥 Demografski podaci za učesnike")
    print(f"  🔄 Veze sličnosti između grupa")
    print(f"  🏷️ Poboljšano označavanje i tipiziranje veza")
    print(f"  ↔️ Eksplicitni Directed/Undirected tipovi veza")

if __name__ == "__main__":
    enhance_gephi_files()
