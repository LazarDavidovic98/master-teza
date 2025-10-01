# 🕸️ BIPARTITNI GRAF I PROJEKCIJE - DOKUMENTACIJA

Ovaj dokument opisuje nove tipove mreža koji su dodani u anketa-sistem za analizu AI upotrebe.

## 📊 TIPOVI GRAFOVA

### 1. 🔄 **Bipartitni (Dvodelni) Graf**
**Fajlovi:** `gephi_nodes_bipartite.csv`, `gephi_edges_bipartite.csv`

Bipartitni graf razdvaja čvorove u **dve odvojene grupe**:
- 👥 **Korisnici** (leva strana, X: -150 do -50)
- 🤖 **AI alati** (desna strana, X: 50 do 150)

**Karakteristike:**
- Veze postoje **SAMO između korisnika i alata** (koristi alat)
- **NEMA veza** korisnik-korisnik ili alat-alat
- Idealan za **collaborative filtering** i **preporuke**

**Atributi čvorova:**
```
Korisnici: Id, Label, Type=user, Partition=users, CiljanaGrupa, Country, Education, ToolsUsed, Size, Color, Shape=circle
AI alati: Id, Label, Type=ai_tool, Partition=tools, Category, UsersCount, Size, Color, Shape=square
```

**Atributi veza:**
```
Source, Target, Weight, Context=tool_usage, Label=koristi, Color=#00CEC9, Purposes
```

### 2. 👥 **User-User Projekcija**
**Fajlovi:** `gephi_nodes_user_projection.csv`, `gephi_edges_user_projection.csv`

Projekcija bipartitnog grafa na **korisnike**:
- Čvorovi: samo korisnici
- Veze: korisnici povezani preko **zajedničkih alata**

**Jaccard sličnost:**
```
similarity = |common_tools| / |all_tools_union|
```

**Pragovi veza:**
- Minimum sličnost: 0.15
- Weight = Jaccard sličnost

### 3. 🤖 **Tool-Tool Projekcija**
**Fajlovi:** `gephi_nodes_tool_projection.csv`, `gephi_edges_tool_projection.csv`

Projekcija bipartitnog grafa na **alate**:
- Čvorovi: samo AI alati
- Veze: alati povezani preko **zajedničkih korisnika**

**Pragovi veza:**
- Minimum sličnost: 0.1
- Weight = Jaccard sličnost

## 🎯 ANALITIČKE MOGUĆNOSTI

### 📈 Bipartitni Graf Analize

1. **Collaborative Filtering**
   - Preporučivanje alata na osnovu sličnih korisnika
   - "Korisnici slični vama takođe koriste..."

2. **Centrality Measures**
   - Degree centrality: najaktivniji korisnici/najpopularniji alati
   - Betweenness: "most brokerage" čvorovi

3. **Bipartite Density**
   ```
   density = edges / (users × tools)
   ```

### 👥 User Projection Analize

1. **Community Detection**
   - Modularity algoritam za grupe korisnika
   - Identifikacija "technology tribes"

2. **Clustering Analysis**
   - Clustering coefficient
   - K-means na user-tool vektorima

3. **Influence Analysis**
   - PageRank na user network-u
   - Identifikacija "AI influencer-a"

### 🤖 Tool Projection Analize

1. **Market Basket Analysis**
   - Koju kombinaciju alata korisnici često koriste zajedno
   - Association rules mining

2. **Tool Clustering**
   - Grupiranje sličnih alata
   - Technology stack prepoznavanje

3. **Recommendation Systems**
   - "Ako koristiš X i Y, možda ti treba i Z"
   - Cross-selling analysis

## 🎨 GEPHI SETUP

### Bipartitni Graf
```
Layout: ForceAtlas 2 (Dissuade Hubs: true)
Node Color: Partition → 'Partition' attribute
Node Size: 'Size' attribute
Edge Color: #00CEC9 (unified teal)
```

### User Projection
```
Layout: ForceAtlas 2 
Node Color: Partition → 'CiljanaGrupa' attribute
Node Size: 'ToolsCount' attribute
Edge Thickness: 'Weight' attribute
Community Detection: Modularity algorithm
```

### Tool Projection
```
Layout: Yifan Hu
Node Color: Partition → 'Category' attribute  
Node Size: 'UsersCount' attribute
Edge Color: #e74c3c (red)
Community Detection: Modularity algorithm
```

## 📊 STATISTIKE PRIMERA

### Bipartitni Graf
- **👥 Korisnici:** 22 (6 grupa)
- **🤖 Alati:** 10 (5 kategorija)
- **🔗 Veze:** 251
- **📈 Gustina:** 1.141

### User Projekcija
- **👥 Čvorovi:** 15 korisnika
- **🔗 Veze:** 69 
- **📈 Gustina:** 0.657
- **🎯 Prosečna sličnost:** 0.370

### Tool Projekcija  
- **🤖 Čvorovi:** 10 alata
- **🔗 Veze:** 42
- **📈 Gustina:** 0.933
- **🎯 Prosečna sličnost:** 0.324

## 🚀 POKRETANJE

```bash
# Generiši bipartitni graf
python generate_bipartite_gephi.py

# Generiši projekcije
python generate_projection_gephi.py
```

## 📁 IZLAZNI FAJLOVI

| Fajl | Opis |
|------|------|
| `gephi_nodes_bipartite.csv` | Bipartitni čvorovi (users + tools) |
| `gephi_edges_bipartite.csv` | Bipartitne veze (user→tool) |
| `gephi_nodes_user_projection.csv` | User čvorovi |
| `gephi_edges_user_projection.csv` | User-user veze |
| `gephi_nodes_tool_projection.csv` | Tool čvorovi |
| `gephi_edges_tool_projection.csv` | Tool-tool veze |
| `network_summary_bipartite.json` | Statistike bipartitnog grafa |
| `network_summary_projections.json` | Statistike projekcija |

## 💡 USE CASES

### Business Intelligence
1. **Tool Adoption Patterns**
   - Koji alati se usvajaju zajedno?
   - Optimalne putanje za onboarding

2. **User Segmentation**
   - Klasterovanje korisnika po AI preferencama
   - Targeted marketing campaigns

3. **Product Recommendations**
   - Algoritmi za preporučivanje novih alata
   - Cross-selling strategies

### Academic Research
1. **Technology Diffusion**
   - Kako se AI alati šire kroz različite grupe
   - Innovation adoption curves

2. **Network Effects**
   - Social influence na AI adopciju
   - Peer effects analysis

3. **Digital Divide**
   - Razlike u AI korišćenju između grupa
   - Access equality analysis

## ⚠️ NAPOMENE

- **Jaccard similarity** je robusnija od Cosine similarity za binarne podatke
- **Bipartite density > 1** označava visoku povezanost (više svrha po alatu)
- **Projection graphs** gube informaciju - koristiti pažljivo
- **Community detection** radi najbolje na projektovanim grafovima

## 🔧 TROUBLESHOOTING

**Problem:** Nema veza u projekciji
**Rešenje:** Smanji prag sličnosti (trenutno 0.15 za users, 0.1 za tools)

**Problem:** Previše veza
**Rešenje:** Povećaj prag sličnosti ili koristi top-k connections

**Problem:** Overlapping čvorovi u Gephi
**Rešenje:** Koristi X,Y koordinate iz CSV-a ili ForceAtlas2 layout
