# Mini-anketni sistem za ispitivanje korišćenja AI alata

**Anonimna** web aplikacija za prikupljanje i analizu odgovora studenata o korišćenju ChatGPT-a i GitHub Copilot-a.

## Funkcionalnosti

- 📝 **Anonimna anketa** - Bez čuvanja ličnih podataka
- 🔗 **Deljenje linka** - Jednostavno slanje ankete drugim osobama
- 💾 **Skladištenje podataka** - Čuvanje odgovora u CSV formatu
- 📊 **Pregled rezultata** - Osnovna statistika i pregled svih odgovora
- 🎨 **Responzivni dizajn** - Prilagođen različitim veličinama ekrana

## Pokretanje aplikacije

### 1. Instaliranje zavisnosti

```powershell
# Navigacija u folder projekta
cd "c:\Users\Administrator\Desktop\master-teza\anketa-sistem"

# Kreiranje virtuelnog okruženja (opciono)
python -m venv venv
.\venv\Scripts\activate

# Instaliranje Flask-a
pip install -r requirements.txt
```

### 2. Pokretanje servera

```powershell
python app.py
```

### 3. Pristup aplikaciji

Otvorite web browser i idite na: http://127.0.0.1:5000

## Struktura aplikacije

```
anketa-sistem/
├── app.py                 # Glavna Flask aplikacija
├── requirements.txt       # Python zavisnosti
├── survey_responses.csv   # Skladište podataka (kreira se automatski)
├── templates/
│   ├── survey.html       # Forma za anketu
│   ├── success.html      # Stranica potvrde
│   └── results.html      # Prikaz rezultata
└── static/               # Statički fajlovi (CSS, JS, slike)
```

## Stranice aplikacije

### 1. Anketa (/)
- **Anonimna anketa** sa pitanjima o:
  - Osnovnim podacima (godina studija, smer - opciono)
  - Korišćenju ChatGPT-a
  - Korišćenju GitHub Copilot-a
  - Opštim stavovima o AI alatima
- **Nema obaveznih ličnih podataka** - potpuno anonimno

### 2. Potvrda (/success)
- Stranica koja se prikazuje nakon uspešnog slanja ankete
- Opcije za novu anketu ili pregled rezultata

### 3. Rezultati (/results)
- Osnovna statistika (broj odgovora, procenat korisnika)
- Tabela sa svim odgovorima
- Mogućnost export-a podataka

## Podaci koji se prikupljaju

- **Osnovni podaci**: Godina studija, smer (opciono)
- **ChatGPT**: Korišćenje, učestalost, svrha
- **Copilot**: Korišćenje, učestalost, svrha  
- **Opšti stavovi**: Uticaj na učenje, prednosti, nedostaci, preporuke

**Napomena**: Anketa je potpuno anonimna - ne čuva se ime, prezime ili bilo koji lični podatak.

## Bezbednost i privatnost

- 🔒 **Potpuno anonimno** - Ne čuva se ime, prezime ili lični podaci
- 💾 **Lokalno skladište** - CSV fajl se čuva lokalno kod vas
- 🔗 **Deljenje linka** - Možete bezbedno deliti link sa studentima
- 🛡️ **Nema baze podataka** - Jednostavna i bezbedna implementacija
- 🔑 **Preporučuje se menjanje secret_key u produkciji**

## Proširenja

Moguća poboljšanja:
- Export u Excel format
- Grafički prikaz statistika
- Filtriranje rezultata
- Admin panel za upravljanje
- Baza podataka umesto CSV
- Autentifikacija korisnika

## Tehnologije

- **Backend**: Python Flask
- **Frontend**: HTML, CSS (vanilla)
- **Skladište**: CSV fajlovi
- **Hosting**: Lokalni server (može se deploy-ovati na cloud)

## Licenca

Projekat je kreiran za edukacione svrhe.
