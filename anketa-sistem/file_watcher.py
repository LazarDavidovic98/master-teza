"""
File Watcher za automatsko ažuriranje Data Science fajlova
Prati promene u survey_responses.csv i automatski regeneriše sve fajlove
"""

import os
import time
import threading
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SurveyFileHandler(FileSystemEventHandler):
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.survey_file = "survey_responses.csv"
        self.last_modified = 0
        self.processing = False
        
    def on_modified(self, event):
        """Poziva se kada se fajl promeni"""
        if event.is_directory:
            return
            
        # Proverava da li je survey_responses.csv promenjen
        if os.path.basename(event.src_path) == self.survey_file:
            current_time = time.time()
            
            # Izbegava multiple pozive (debouncing) - povećao vreme sa 2 na 5 sekundi
            if current_time - self.last_modified < 5:
                return
                
            if self.processing:
                print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] Već se obrađuje - preskačem")
                return
                
            self.last_modified = current_time
            self.processing = True
            
            print(f"\n🔄 [{datetime.now().strftime('%H:%M:%S')}] Detektovana promena u {self.survey_file}")
            
            # Pokreni regeneraciju u background thread-u
            threading.Thread(target=self.regenerate_files, daemon=True).start()
    
    def regenerate_files(self):
        """Regeneriši sve data science fajlove"""
        try:
            print("📊 Regenerišem Data Science fajlove...")
            
            # Obradi postojeće podatke
            processed_count = self.data_manager.process_existing_survey_data(self.survey_file)
            
            if processed_count > 0:
                # Kreiraj network graf
                G = self.data_manager.create_network_graph()
                nodes = G.number_of_nodes()
                edges = G.number_of_edges()
                
                # Generiši Gephi fajlove
                self.data_manager.generate_gephi_files()
                
                print(f"✅ Uspešno ažurirano!")
                print(f"   • {processed_count} učesnika obrađeno")
                print(f"   • Network: {nodes} čvorova, {edges} veza")
                print(f"   • Gephi fajlovi regenerisani")
            else:
                print("⚠️ Nema podataka za obradu")
                
        except Exception as e:
            print(f"❌ Greška pri regeneraciji: {e}")
        finally:
            self.processing = False

class FileWatcher:
    def __init__(self, data_manager, watch_directory="."):
        self.data_manager = data_manager
        self.watch_directory = watch_directory
        self.observer = None
        self.handler = SurveyFileHandler(data_manager)
        self.running = False
        
    def start(self):
        """Pokreni file watcher"""
        if self.running:
            return
            
        try:
            self.observer = Observer()
            self.observer.schedule(self.handler, self.watch_directory, recursive=False)
            self.observer.start()
            self.running = True
            print(f"👁️ File Watcher pokrenut - prati promene u survey_responses.csv")
            
        except Exception as e:
            print(f"❌ Greška pri pokretanju File Watcher-a: {e}")
    
    def stop(self):
        """Zaustavi file watcher"""
        if not self.running or not self.observer:
            return
            
        try:
            self.observer.stop()
            self.observer.join()
            self.running = False
            print("🛑 File Watcher zaustavljen")
            
        except Exception as e:
            print(f"❌ Greška pri zaustavljanju File Watcher-a: {e}")
    
    def is_running(self):
        """Proverava da li je watcher pokrenut"""
        return self.running and self.observer and self.observer.is_alive()
