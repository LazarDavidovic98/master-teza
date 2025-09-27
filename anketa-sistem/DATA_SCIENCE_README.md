# 🔬 Data Science Extension za Anketa Sistem

Ovo je napredna ekstenzija koja omogućava kreiranje povezanih CSV fajlova i sprovođenje data science analiza preko Pandas, NetworkX i Gephi alata.

## 🚀 Funkcionalnosti

### 📊 Povezani CSV fajlovi
Sistem automatski kreira sledeće strukturirane fajlove:

- **`participants.csv`** - Demografski podaci učesnika
- **`ai_knowledge.csv`** - AI znanje po učesniku i alatu  
- **`quiz_responses.csv`** - Individualni odgovori na kviz pitanja
- **`relationships.csv`** - Veze između učesnika i AI alata (za NetworkX)
- **`tool_usage_patterns.csv`** - Patterns korišćenja alata
- **`gephi_nodes.csv` & `gephi_edges.csv`** - Fajlovi za Gephi import

### 🔗 Tipovi veza za network analizu
- **uses_tool** - Učesnik koristi AI alat
- **affiliated_with** - Učesnik povezan sa organizacijom
- **similar_usage** - Slični patterns korišćenja

### 🎯 Tipovi korisnika na osnovu usage patterns
- **focused_user** - Koristi jedan alat za jednu svrhu
- **power_user** - Koristi mnogo alata za mnogo svrha
- **tool_explorer** - Više alata nego svrha
- **purpose_diverse** - Više svrha nego alata
- **balanced_user** - Balansiran pristup

## 📦 Instalacija

### Korak 1: Instaliraj potrebne biblioteke
```bash
pip install -r requirements_data_science.txt
```

### Korak 2: Generiši test podatke (opciono)
```bash
# Za kompleksnu analizu sa 50-100 učesnika:
python demo_data_science.py

# Za mass testiranje sa 300+ novih redova:
python generate_mass_data.py
```

### Korak 3: Pokreni Flask aplikaciju
```bash
python app.py
```

### Korak 4: Idi na Data Science Dashboard
Otvori: `http://127.0.0.1:5000/data_science`

## 🎯 Test Data Generatori

### `demo_data_science.py` - Kompletan sistem
- Generiše **50-100 učesnika** sa kompletnom analizom
- Automatski pokreće napredne analize
- Idealno za development i testing
- Kreira sve povezane CSV fajlove

### `generate_mass_data.py` - Mass Data Generator
- Generiše **300+ novih redova** direktno u survey_responses.csv
- Dodaje podatke u postojeći fajl (ne briše stare)
- Realistički podaci sa 70% tačnih quiz odgovora
- File watcher automatski regeneriše sve analize
- Idealno za stress testing i performance analizu

## 🔍 Analize koje možete uraditi

### 1. **Pandas Analize**
```python
from advanced_analytics import AdvancedAnalytics
analytics = AdvancedAnalytics()

# Demografska analiza
analytics.demographic_analysis()

# Clustering korisnika 
tool_matrix, clusters = analytics.ai_tool_clustering()

# Korelacije između varijabli
correlation_matrix, merged_data = analytics.correlation_analysis()
```

### 2. **NetworkX Analize** 
```python
from data_science_extension import DataScienceManager
import networkx as nx

data_manager = DataScienceManager()
G = data_manager.create_network_graph()

# Centralnost mere
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Community detection
communities = list(nx.community.louvain_communities(G))
print(f"Pronađeno {len(communities)} zajednica")
```

### 3. **Gephi Vizualizacije**
1. Pokreni Flask aplikaciju
2. Idi na `/data_science` rutu  
3. Klikni "Generate Gephi Files"
4. Otvori Gephi
5. Import `data/exports/gephi_edges.csv` kao edges
6. Import `data/exports/gephi_nodes.csv` kao nodes
7. Primeni layout algoritam (npr. ForceAtlas2)
8. Eksperimentiraj sa bojama i veličinama

## 📈 Primer korišćenja

```python
# 1. Kreiraj Data Science Manager
from data_science_extension import DataScienceManager
data_manager = DataScienceManager()

# 2. Obradi survey odgovor (automatski se poziva u Flask app)
participant_id = data_manager.process_survey_response(survey_data)

# 3. Generiši network graf
G = data_manager.create_network_graph()

# 4. Eksportuj za Gephi
nodes_file, edges_file = data_manager.generate_gephi_files()

# 5. Pokreni napredne analize
from advanced_analytics import AdvancedAnalytics
analytics = AdvancedAnalytics()
analytics.generate_insights()
```

## 🗂️ Struktura podataka

### Participants CSV
```csv
participant_id,timestamp,birth_year,country,education,employment_status,age_group,experience_level
P12AB34CD,2025-09-26 10:30:00,1995,Serbia,master,employed,mid_career,intermediate
```

### AI Knowledge CSV  
```csv
participant_id,ai_tool,knowledge_level,usage_frequency,purpose,effectiveness_rating,tool_category
P12AB34CD,ChatGPT,4,daily,code_generation,4,conversational_ai
P12AB34CD,GitHub Copilot,3,weekly,debugging,3,code_assistant
```

### Quiz Responses CSV
```csv
participant_id,question_id,question_category,given_answer,correct_answer,is_correct,difficulty_level
P12AB34CD,chatgpt_omni,technical,GPT-4,GPT-4,true,medium
P12AB34CD,creativity_parameter,conceptual,Temperature,Temperature,true,hard
```

### Relationships CSV (za NetworkX)
```csv
source_id,target_id,relationship_type,weight,context
P12AB34CD,TOOL_ChatGPT,uses_tool,0.8,ai_tool_usage
P12AB34CD,ORG_University_Belgrade,affiliated_with,1.0,organizational
```

## 🎯 Use Cases

### 1. **Akademska istraživanja**
- Korelacija između demografije i AI korišćenja
- Cross-tabulation analiza
- Statistical significance testing

### 2. **Industrijaka primena**
- User segmentation na osnovu AI tool preferencija  
- Network effects u AI adoption
- Predictive modeling za future AI usage

### 3. **Edukacija**
- Identifikacija knowledge gaps
- Personalizovane preporuke za učenje
- Community-based learning insights

## 🛠️ Troubleshooting

### Problem: "Data Science extension not available"
**Rešenje:** Instaliraj potrebne biblioteke:
```bash
pip install networkx scikit-learn
```

### Problem: Nema podataka za analizu
**Rešenje:** Pokreni demo script:
```bash
python demo_data_science.py
```

### Problem: Gephi import greške
**Rešenje:** 
1. Uveri se da su fajlovi kreirani (`/data_science` ruta)
2. U Gephi import edges fajl prvo, zatim nodes
3. Proveri da li su kolone pravilno mapirane

## 📊 Primer NetworkX analize

```python
import networkx as nx
import matplotlib.pyplot as plt
from data_science_extension import DataScienceManager

# Kreiraj graf
data_manager = DataScienceManager() 
G = data_manager.create_network_graph()

# Centrality analize
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Najuticajniji čvorovi  
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 najuticajnijih čvorova:")
for node, centrality in top_nodes:
    print(f"{node}: {centrality:.3f}")

# Community detection
communities = list(nx.community.louvain_communities(G))
print(f"Broj zajednica: {len(communities)}")

# Vizualizacija (osnovna)
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=300, font_size=8)
plt.title("AI Tools Usage Network")
plt.show()
```

## 🎨 Gephi Workflow

1. **Import podataka:**
   - File → Import Spreadsheet → `gephi_edges.csv`
   - Izaberi "Edges table"
   - File → Import Spreadsheet → `gephi_nodes.csv` 
   - Izaberi "Nodes table"

2. **Layout algoritmi:**
   - **ForceAtlas2** - Za opštu strukturu
   - **Fruchterman Reingold** - Za kompaktne vizualizacije  
   - **Circular** - Za hijerarhijske strukture

3. **Stilizovanje:**
   - Node size → Degree centrality
   - Node color → Type (participant vs tool)
   - Edge weight → Relationship strength

4. **Filtriranje:**
   - Topology → Giant Component (prikaži glavnu komponentu)
   - Attributes → Filter po kategorijama

## 📚 Dodatni resursi

- [NetworkX dokumentacija](https://networkx.org/documentation/stable/)
- [Gephi tutorials](https://gephi.org/tutorials/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html)

---

**Autor:** AI Anketa Sistem  
**Verzija:** 1.0  
**Datum:** Septembar 2025
