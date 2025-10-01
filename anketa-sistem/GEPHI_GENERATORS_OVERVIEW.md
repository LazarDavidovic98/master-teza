# 📊 PREGLED SVIH GEPHI GENERATORA

Ovaj dokument daje pregled svih dostupnih skriptova za generisanje mrežnih analiza.

## 🎯 GLAVNI GENERATORI

### 1. **`generate_gephi_files.py`** - Osnovni Graf  
```bash
python generate_gephi_files.py
```
**Kreira:**
- `data/gephi_nodes.csv` - Osnovni čvorovi
- `data/gephi_edges.csv` - Osnovne veze  

**Tipovi čvorova:** participants, target_groups, ai_tools, organizations

---

### 2. **`generate_enhanced_network.py`** - Unapređeni Graf ⭐
```bash  
python generate_enhanced_network.py
```
**Kreira:**
- `gephi_nodes_enhanced.csv` - Čvorovi sa dodatnim atributima
- `gephi_edges_enhanced.csv` - Veze sa sličnostima između grupa
- `network_summary_enhanced.json` - Statistike

**Novo u ovoj verziji:**
- 🎨 Shape atributi (circle, diamond, square, triangle)
- 📍 X,Y koordinate za početni layout  
- 👥 Demografski podaci za učesnike
- 🔄 Veze sličnosti između grupa
- ↔️ Directed/Undirected tipovi veza

---

### 3. **`create_group_focused_network.py`** - Grupa-Fokusirani Graf
```bash
python create_group_focused_network.py  
```
**Kreira:**
- `gephi_nodes_group_focused.csv` - Fokus na grupe
- `gephi_edges_group_focused.csv` - Inter-grupne veze

---

### 4. **`generate_bipartite_gephi.py`** - Bipartitni Graf 🕸️
```bash
python generate_bipartite_gephi.py
```
**Kreira:**
- `gephi_nodes_bipartite.csv` - Korisnici (levo) i alati (desno)
- `gephi_edges_bipartite.csv` - Samo user→tool veze
- `network_summary_bipartite.json` - Statistike

**Specijalno:** Dvodelni graf idealan za collaborative filtering

---

### 5. **`generate_projection_gephi.py`** - Projekcije Bipartitnog Grafa
```bash  
python generate_projection_gephi.py
```
**Kreira:**
- `gephi_nodes_user_projection.csv` - Korisnici povezani preko alata
- `gephi_edges_user_projection.csv` - User-user veze (Jaccard similarity)
- `gephi_nodes_tool_projection.csv` - Alati povezani preko korisnika  
- `gephi_edges_tool_projection.csv` - Tool-tool veze
- `network_summary_projections.json` - Statistike

---

## 🚀 BRZI START

### Generiši sve tipove grafova odjednom:
```bash
# Osnovni i unapređeni
python generate_gephi_files.py
python generate_enhanced_network.py

# Bipartitni grafovi (NOVO!)
python generate_bipartite_gephi.py
python generate_projection_gephi.py

# Grupa-fokusiran (opcionalno)  
python create_group_focused_network.py
```

### Ili koristi Web Interface:
1. Pokrete aplikaciju: `python app.py`
2. Idite na: `http://127.0.0.1:5000/data_science`
3. Kliknite dugmićima:
   - 🔄 "Regeneriši Gephi Fajlove" (osnovni)
   - 🕸️ "Regeneriši Dvodelne Grafove" (bipartitni + projekcije)

---

## 📁 IZLAZNI FAJLOVI

| Generator | Nodes | Edges | Summary |
|-----------|-------|-------|---------|
| Osnovni | `data/gephi_nodes.csv` | `data/gephi_edges.csv` | `network_summary.json` |
| Unapređeni | `gephi_nodes_enhanced.csv` | `gephi_edges_enhanced.csv` | `network_summary_enhanced.json` |
| Grupa-fokus | `gephi_nodes_group_focused.csv` | `gephi_edges_group_focused.csv` | `network_summary_group_focused.json` |
| Bipartitni | `gephi_nodes_bipartite.csv` | `gephi_edges_bipartite.csv` | `network_summary_bipartite.json` |
| User projekcija | `gephi_nodes_user_projection.csv` | `gephi_edges_user_projection.csv` | `network_summary_projections.json` |
| Tool projekcija | `gephi_nodes_tool_projection.csv` | `gephi_edges_tool_projection.csv` | ↗️ isti kao user |

---

## 💡 PREPORUKE ZA GEPHI

### Za Osnovnu Analizu:
**Koristite:** `generate_enhanced_network.py`
- Najbolji balans funkcionalnosti i jednostavnosti
- Gotove X,Y koordinate  
- Više atributa za stilizovanje

### Za Recommendation Systems:
**Koristite:** Bipartitni grafovi
- `generate_bipartite_gephi.py` → collaborative filtering
- `generate_projection_gephi.py` → user/tool sličnosti

### Za Community Detection:
**Koristite:** User/Tool projekcije
- Modularity algoritam najbolje radi na projektovanim grafovima

---

## 📋 REDOSLED IZVRŠAVANJA

**Važno:** Uvek prvo izvršite `process_real_data.py` ako se `survey_responses.csv` promenio!

```bash
# 1. Obradi sirove podatke (ako potrebno)
python process_real_data.py

# 2. Generiši željene tipove grafova
python generate_enhanced_network.py        # Preporučeno
python generate_bipartite_gephi.py          # Za naprednu analizu  
python generate_projection_gephi.py         # Za collaborative filtering
```

---

## 🔄 HISTORY

- **v1.0:** `generate_gephi_files.py` - osnovni graf
- **v1.1:** `generate_gephi_enhanced.py` - prva unapređenja
- **v2.0:** `generate_enhanced_network.py` - kompletno remaster ⭐  
- **v2.1:** `generate_bipartite_gephi.py` - bipartitni grafovi
- **v2.2:** `generate_projection_gephi.py` - projekcije i sličnosti

📖 **Detaljnu dokumentaciju o bipartitnim grafovima** potražite u `BIPARTITE_GRAPHS_README.md`
