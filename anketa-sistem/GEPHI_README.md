# Gephi Network Files - Analiza mreže na osnovu ciljanih grupa

## 📊 Generisani fajlovi

### 1. Osnovni Gephi fajlovi
- **`gephi_nodes.csv`** - Osnovni čvorovi (51 čvorova)
- **`gephi_edges.csv`** - Osnovne veze (292 veze)

### 2. Poboljšani Gephi fajlovi sa bojama i oblicima
- **`gephi_nodes_enhanced.csv`** - Čvorovi sa bojama, oblicima i veličinama
- **`gephi_edges_enhanced.csv`** - Veze sa bojama i debljinom linija

### 3. Gephi fajlovi fokusirani na ciljane grupe ⭐ **PREPORUČENO**
- **`gephi_nodes_group_focused.csv`** - Optimizovano za analizu grupa (40 čvorova)
- **`gephi_edges_group_focused.csv`** - Veze fokusirane na grupe (122 veze)

### 4. Dodatni fajlovi
- **`network_summary.json`** - Statistike mreže
- **`generate_gephi_files.py`** - Script za osnovne fajlove
- **`generate_gephi_enhanced.py`** - Script za poboljšane fajlove  
- **`create_group_focused_network.py`** - Script za grup-fokusirane fajlove

## 🎯 Ciljane grupe kao centralni čvorovi

### Analizirane grupe:
1. **PROSVETA** (12 članova) - Najčešće: Crna Gora, Visa, 20+ godina iskustva
2. **IT_INDUSTRIJA** (9 članova) - Najčešće: Nemačka, Doktorat, 16-20 godina iskustva  
3. **MEDICINA** (12 članova) - Najčešće: Srbija, Srednja, 6-10 godina iskustva
4. **KREATIVNA_FILMSKA** (8 članova) - Najčešće: Bugarska, Visoka, 6-10 godina iskustva
5. **DRUSTVENE_NAUKE** (5 članova) - Najčešće: Nemačka, Doktorat, 0 godina iskustva
6. **OSTALO** (4 članova) - Najčešće: Slovenija, Master, 0 godina iskustva

## 🔗 Tipovi veza u mreži

### Glavne veze:
- **belongs_to / pripada** - Učesnik pripada ciljanoj grupi (50 veza)
- **used_by_group** - AI alat se koristi od strane grupe (57 veza)  
- **similar_groups** - Grupe su slične po karakteristikama (15 veza)

### Analizirane sličnosti među grupama:
- **Prosveta ↔ Medicina**: 100% sličnost (najjača veza)
- **IT Industrija ↔ Medicina**: 90% sličnost
- **Kreativna/Filmska ↔ Medicina**: 100% sličnost
- **Prosveta ↔ IT Industrija**: 85% sličnost

## 📈 Struktura mreže

### Čvorovi po tipovima:
- **🔴 Ciljane grupe** (6 čvorova) - Dijamant oblik, crvena boja
- **🔵 Učesnici** (24 čvora) - Krug oblik, plava boja  
- **🟢 AI alati** (10 čvorova) - Kvadrat oblik, zelena boja
- **🟠 Organizacije** (11 čvorova) - Trougao oblik, narandžasta boja

### AI alati u mreži:
- ChatGPT, Claude, Gemini (Conversational AI)
- GitHub Copilot (Code Assistant)
- Midjourney, DALL-E, Stable Diffusion (Image Generation)
- Grammarly (Writing Assistant)
- Bing Chat (Search Assistant)
- DeepL (Translation)

## 🚀 Kako koristiti u Gephi

### Za najbolje rezultate, koristite **group_focused** fajlove:

1. **Otvorite Gephi**
2. **Importujte čvorove**: File → Import Spreadsheet → `gephi_nodes_group_focused.csv`
3. **Importujte veze**: File → Import Spreadsheet → `gephi_edges_group_focused.csv`

### Preporučena podešavanja u Gephi:

#### Layout algoritmi:
- **ForceAtlas 2** - Za grupisanje po sličnosti
- **Fruchterman Reingold** - Za jasno razdvajanje grupa
- **Yifan Hu** - Za kompaktnu strukturu

#### Vizualizacija:
- **Veličina čvorova**: Koristite `Size` kolonu
- **Boje čvorova**: Koristite `Color` kolonu
- **Oblik čvorova**: Koristite `Shape` kolonu  
- **Debljina veza**: Koristite `Thickness` kolonu

#### Filteri:
- Filtrirajte po `Type` za fokus na određene tipove čvorova
- Filtrirajte po `Group_Name` za analizu pojedinačnih grupa
- Filtrirajte po `Relationship` za različite tipove veza

## 📊 Ključni uvidi iz mreže

### 1. Centriranost grupa
Sve ciljane grupe su pozicionirane centralno sa direktnim vezama ka svojim članovima.

### 2. AI alati kao povezivači
AI alati služe kao mostovi između različitih grupa, omogućavajući analizu zajedničkih interesovanja.

### 3. Sličnost grupa
Visoka sličnost između grupa ukazuje na zajedničke obrasce korišćenja AI alata i demografske karakteristike.

### 4. Organizaciona povezanost  
Institucije su povezane sa različitim grupama, prikazujući međusektorsku saradnju.

## 🔧 Tehnički detalji

### Struktura podataka:
- **Izvorni podaci**: `survey_responses.csv`
- **Procesiranje**: Python pandas, numpy
- **Format**: Gephi-kompatibilni CSV fajlovi
- **Encoding**: UTF-8

### Algoritmi korišćeni:
- Jaccard similarity za grupe
- Demografska analiza za učesnike  
- Usage pattern analiza za AI alate
- Network centrality measures

---

**Napomena**: Za detaljnu analizu preporučuje se korišćenje `group_focused` fajlova jer su optimizovani za istraživanje veza između ciljanih grupa i omogućavaju najjasniju vizualizaciju mrežne strukture.
