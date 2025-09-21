# 🚀 Kako pokrenuti i zaustaviti anketu

## ✅ POKRETANJE ANKETE

### 1. Otvorite PowerShell kao Administrator
- Kliknite desnim klikom na Start dugme
- Izaberite "Windows PowerShell (Admin)"

### 2. Idite u folder ankete
```powershell
cd "C:\Users\Administrator\Desktop\master-teza\anketa-sistem"
```

### 3. Pokrenite Flask aplikaciju (Terminal 1)
```powershell
python app.py
```
**OSTAVITE OVAJ TERMINAL OTVOREN!**

### 4. Otvorite novi PowerShell terminal
- Ponovo otvorite PowerShell
- Idite u isti folder:
```powershell
cd "C:\Users\Administrator\Desktop\master-teza\anketa-sistem"
```

### 5. Pokrenite ngrok (Terminal 2)
```powershell
.\ngrok.exe http 5000
```
**OSTAVITE I OVAJ TERMINAL OTVOREN!**

### 6. Kopirajte link
- Ngrok će pokazati URL kao: `https://abc123.ngrok-free.app`
- **OVAJ LINK podelite sa ljudima!**

---

## ❌ ZAUSTAVLJANJE ANKETE

### Opcija 1: Zatvaranje terminala
- Zatvorite oba PowerShell prozora
- Anketa će se automatski zaustaviti

### Opcija 2: Ctrl+C
- U svakom terminalu pritisnite `Ctrl+C`
- Potvrdite sa `Y` ako pita

### Opcija 3: Preko Task Manager-a
```powershell
# Zaustavite Flask
taskkill /f /im python.exe

# Zaustavite ngrok
taskkill /f /im ngrok.exe
```

---

## 🔄 PONOVO POKRETANJE

Ako se anketa zaustavila:

1. **Ponovite korake 1-5 odozgo**
2. **VAŽNO**: Link će se promeniti! 
3. **Ažurirajte link** u success.html fajlu:
   - Otvorite: `templates/success.html`
   - Pronađite: `<div class="link-box">`
   - Zamenite stari link sa novim ngrok linkom

---

## 📊 PRISTUP REZULTATIMA

- Lokalno: http://127.0.0.1:5000/results
- Online: https://VAŠ-NGROK-LINK.ngrok-free.app/results

---

## ⚠️ VAŽNE NAPOMENE

- **Oba terminala moraju da budu otvorena** da bi anketa radila
- **Svaki put kada restartujete ngrok, dobijate novi link**
- **Rezultati se čuvaju u `survey_responses.csv`** - neće se izgubiti
- **Za trajno hosting** koristite Railway, Heroku ili PythonAnywhere

---

## 🆘 ČESTI PROBLEMI

### Problem: "python nije prepoznat"
**Rešenje:** Instalirajte Python ili dodajte ga u PATH

### Problem: "ngrok authtoken greška"  
**Rešenje:**
```powershell
.\ngrok.exe config add-authtoken 331e85XBvVf11uWNadSBFBb4BPK_3wbQhvpnGEMLeR6EtE6nW
```

### Problem: "Port 5000 je zauzet"
**Rešenje:**
```powershell
# Zaustavite sve Python procese
taskkill /f /im python.exe
```

### Problem: "Više ngrok sesija"
**Rešenje:**
```powershell
# Zaustavite sve ngrok procese
taskkill /f /im ngrok.exe
```

---

## 📱 DELJENJE ANKETE

**Pošaljite ovakvu poruku:**

```
Pozdrav!

Molim vas da popunite kratku anonimnu anketu o korišćenju AI alata (ChatGPT, Copilot):

https://VAŠ-NGROK-LINK.ngrok-free.app

Anketa traje 3-5 minuta i potpuno je anonimna.
Hvala!

Lazar Davidović
```

---

## 🎯 BRZE KOMANDE

**Pokretanje:**
```powershell
cd "C:\Users\Administrator\Desktop\master-teza\anketa-sistem"
python app.py
# U novom terminalu:
.\ngrok.exe http 5000
```

**Zaustavljanje:**
```powershell
Ctrl+C (u oba terminala)
```
