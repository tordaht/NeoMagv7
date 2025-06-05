# NeoMag V7 - Debug Analysis & Integration Test
# TabPFN & Gemini AI Real Integration Check

import sys
import time
import requests
import json
import pandas as pd
from pathlib import Path

class DebugAnalysis:
    """
    TabPFN ve Gemini entegrasyonlarının gerçek durumunu analiz eder
    """
    
    def __init__(self):
        self.results = {
            'tabpfn_status': {},
            'gemini_status': {},
            'simulation_errors': [],
            'integration_issues': [],
            'recommendations': []
        }
        
    def run_debug_analysis(self):
        """Kapsamlı debug analizi çalıştır"""
        print("=" * 60)
        print("🐛 NeoMag V7 DEBUG ANALİZ - TabPFN & Gemini")
        print("=" * 60)
        
        self.check_tabpfn_real_integration()
        self.check_gemini_real_integration()
        self.test_simulation_endpoints()
        self.analyze_csv_data_flow()
        self.test_mock_vs_real_predictions()
        self.generate_debug_report()
        
    def check_tabpfn_real_integration(self):
        """TabPFN gerçek entegrasyon kontrolü"""
        print("\n🔍 1. TabPFN ENTEGRASYON ANALİZİ")
        print("-" * 30)
        
        try:
            # TabPFN modülü import test
            try:
                from ml_models.tabpfn_integration import create_tabpfn_predictor, TABPFN_AVAILABLE
                self.results['tabpfn_status']['module_available'] = True
                self.results['tabpfn_status']['TABPFN_AVAILABLE'] = TABPFN_AVAILABLE
                print(f"  ✅ TabPFN modülü import edildi: TABPFN_AVAILABLE = {TABPFN_AVAILABLE}")
                
                # Predictor oluşturma testi
                predictor = create_tabpfn_predictor("biophysical", device='cpu')
                if predictor is not None:
                    self.results['tabpfn_status']['predictor_created'] = True
                    print(f"  ✅ TabPFN predictor oluşturuldu: {type(predictor)}")
                    
                    # Gerçek tahmin testi
                    try:
                        import numpy as np
                        X_test = np.random.rand(50, 10)
                        y_test = np.random.rand(50)
                        
                        if hasattr(predictor, 'predict_fitness_landscape'):
                            predictions = predictor.predict_fitness_landscape(X_test, y_test, X_test)
                            self.results['tabpfn_status']['real_prediction'] = True
                            print(f"  ✅ Gerçek TabPFN tahmini yapıldı: {len(predictions)} prediction")
                        else:
                            self.results['tabpfn_status']['real_prediction'] = False
                            print(f"  ❌ TabPFN predictor'da predict_fitness_landscape metodu yok")
                    except Exception as e:
                        self.results['tabpfn_status']['real_prediction'] = False
                        print(f"  ❌ TabPFN tahmin hatası: {e}")
                else:
                    self.results['tabpfn_status']['predictor_created'] = False
                    print(f"  ⚠️ TabPFN predictor None döndü - Mock mode")
                    
            except ImportError as e:
                self.results['tabpfn_status']['module_available'] = False
                print(f"  ❌ TabPFN modülü import edilemedi: {e}")
                
        except Exception as e:
            print(f"  ❌ TabPFN test hatası: {e}")
            self.results['simulation_errors'].append(f"TabPFN test: {e}")
    
    def check_gemini_real_integration(self):
        """Gemini AI gerçek entegrasyon kontrolü"""
        print("\n🤖 2. GEMİNİ AI ENTEGRASYON ANALİZİ")
        print("-" * 30)
        
        try:
            from web_server import GeminiAI, GEMINI_API_KEY
            
            # API key kontrolü
            if GEMINI_API_KEY and GEMINI_API_KEY != "":
                self.results['gemini_status']['api_key_present'] = True
                print(f"  ✅ Gemini API Key mevcut: {GEMINI_API_KEY[:20]}...")
            else:
                self.results['gemini_status']['api_key_present'] = False
                print(f"  ❌ Gemini API Key eksik")
                
            # Gemini AI instance test
            gemini = GeminiAI()
            self.results['gemini_status']['instance_created'] = True
            print(f"  ✅ GeminiAI instance oluşturuldu")
            
            # Gerçek API çağrısı testi
            try:
                test_data = {
                    'bacteria_count': 50,
                    'time_step': 100,
                    'avg_fitness': 0.75,
                    'avg_energy': 60.5
                }
                
                response = gemini.analyze_simulation_data(test_data)
                
                if response and response != "AI analizi şu anda mevcut değil":
                    self.results['gemini_status']['real_api_call'] = True
                    print(f"  ✅ Gerçek Gemini API çağrısı başarılı")
                    print(f"  📝 Response preview: {response[:100]}...")
                else:
                    self.results['gemini_status']['real_api_call'] = False
                    print(f"  ❌ Gemini API çağrısı başarısız: {response}")
                    
            except Exception as e:
                self.results['gemini_status']['real_api_call'] = False
                print(f"  ❌ Gemini API test hatası: {e}")
                
        except Exception as e:
            print(f"  ❌ Gemini test hatası: {e}")
            self.results['simulation_errors'].append(f"Gemini test: {e}")
    
    def test_simulation_endpoints(self):
        """Simülasyon endpoint'lerini test et"""
        print("\n🌐 3. SİMÜLASYON ENDPOİNT TESTİ")
        print("-" * 30)
        
        base_url = "http://127.0.0.1:5000"
        endpoints_to_test = [
            ('/api/start_simulation', 'POST'),
            ('/api/trigger_tabpfn_analysis', 'POST'),
            ('/api/ai_analysis', 'POST'),
            ('/api/analyze_tabpfn_results', 'POST'),
            ('/api/simulation_data', 'GET')
        ]
        
        for endpoint, method in endpoints_to_test:
            try:
                if method == 'GET':
                    response = requests.get(f"{base_url}{endpoint}", timeout=5)
                else:
                    response = requests.post(f"{base_url}{endpoint}", 
                                           json={}, 
                                           headers={'Content-Type': 'application/json'},
                                           timeout=5)
                
                if response.status_code == 200:
                    print(f"  ✅ {endpoint}: {response.status_code}")
                elif response.status_code == 415:
                    print(f"  ⚠️ {endpoint}: Content-Type hatası (415)")
                    self.results['integration_issues'].append(f"Content-Type error: {endpoint}")
                else:
                    print(f"  ❌ {endpoint}: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"  🔌 {endpoint}: Server bağlantısı yok")
            except Exception as e:
                print(f"  ❌ {endpoint}: {e}")
    
    def analyze_csv_data_flow(self):
        """CSV veri akışını analiz et"""
        print("\n📊 4. CSV VERİ AKIŞ ANALİZİ")
        print("-" * 30)
        
        # CSV dosyalarının varlığını kontrol et
        csv_files = [
            'data/simulation_data_*.csv',
            'data/tabpfn_results_*.csv'
        ]
        
        data_dir = Path('data')
        if data_dir.exists():
            print(f"  ✅ Data klasörü mevcut")
            
            # Simulation CSV files
            sim_csvs = list(data_dir.glob('simulation_data_*.csv'))
            if sim_csvs:
                latest_sim = max(sim_csvs, key=lambda x: x.stat().st_mtime)
                try:
                    df = pd.read_csv(latest_sim)
                    print(f"  ✅ Simulation CSV: {len(df)} satır, {len(df.columns)} sütun")
                    print(f"      Columns: {list(df.columns)}")
                    self.results['tabpfn_status']['csv_data_available'] = True
                except Exception as e:
                    print(f"  ❌ Simulation CSV okuma hatası: {e}")
            else:
                print(f"  ⚠️ Simulation CSV dosyası bulunamadı")
                
            # TabPFN Results CSV
            tabpfn_csvs = list(data_dir.glob('tabpfn_results_*.csv'))
            if tabpfn_csvs:
                latest_tabpfn = max(tabpfn_csvs, key=lambda x: x.stat().st_mtime)
                try:
                    df = pd.read_csv(latest_tabpfn)
                    print(f"  ✅ TabPFN Results CSV: {len(df)} satır")
                    print(f"      Analysis methods: {df['analysis_method'].unique()}")
                    self.results['tabpfn_status']['results_csv_available'] = True
                except Exception as e:
                    print(f"  ❌ TabPFN Results CSV okuma hatası: {e}")
            else:
                print(f"  ⚠️ TabPFN Results CSV dosyası bulunamadı")
        else:
            print(f"  ❌ Data klasörü mevcut değil")
    
    def test_mock_vs_real_predictions(self):
        """Mock vs Gerçek tahmin karşılaştırması"""
        print("\n🎯 5. MOCK vs GERÇEK TAHMİN ANALİZİ")
        print("-" * 30)
        
        try:
            import web_server
            sim = web_server.NeoMagV7WebSimulation()
            sim.initialize_engines()
            
            # Mock bakteriler oluştur
            print("  🧪 Test simülasyonu başlatılıyor...")
            sim.start_simulation(initial_bacteria=20)
            
            # 5 saniye çalıştır
            time.sleep(3)
            
            # TabPFN analizi tetikle
            try:
                sim._run_tabpfn_analysis()
                print("  ✅ TabPFN analizi tetiklendi")
                
                # Sonuçları kontrol et
                if sim.scientific_data['tabpfn_predictions']:
                    latest_prediction = sim.scientific_data['tabpfn_predictions'][-1]
                    analysis_method = latest_prediction.get('analysis_method', 'Unknown')
                    
                    if "Mock" in analysis_method:
                        print(f"  ⚠️ Mock analiz kullanılıyor: {analysis_method}")
                        self.results['tabpfn_status']['using_mock'] = True
                    elif "Wright-Fisher" in analysis_method:
                        print(f"  ✅ Gerçek Wright-Fisher analizi: {analysis_method}")
                        self.results['tabpfn_status']['using_mock'] = False
                    else:
                        print(f"  ✅ Gerçek TabPFN analizi: {analysis_method}")
                        self.results['tabpfn_status']['using_mock'] = False
                        
                    print(f"      Prediction mean: {latest_prediction.get('predictions_mean', 0):.4f}")
                    print(f"      Prediction std: {latest_prediction.get('predictions_std', 0):.4f}")
                else:
                    print(f"  ❌ TabPFN prediction verisi bulunamadı")
                    
            except Exception as e:
                print(f"  ❌ TabPFN analiz hatası: {e}")
            
            # Simülasyonu durdur
            sim.stop_simulation()
            print("  ✅ Test simülasyonu durduruldu")
            
        except Exception as e:
            print(f"  ❌ Mock vs Real test hatası: {e}")
            self.results['simulation_errors'].append(f"Mock vs Real test: {e}")
    
    def generate_debug_report(self):
        """Debug raporu oluştur"""
        print("\n" + "=" * 60)
        print("📋 DEBUG RAPORU")
        print("=" * 60)
        
        # TabPFN Durumu
        print("\n🔬 TabPFN DURUMU:")
        tabpfn = self.results['tabpfn_status']
        
        if tabpfn.get('module_available'):
            print("  ✅ TabPFN modülü import edilebiliyor")
            if tabpfn.get('TABPFN_AVAILABLE'):
                print("  ✅ TABPFN_AVAILABLE = True")
            else:
                print("  ⚠️ TABPFN_AVAILABLE = False")
                
            if tabpfn.get('predictor_created'):
                print("  ✅ TabPFN predictor oluşturulabiliyor")
                if tabpfn.get('real_prediction'):
                    print("  ✅ Gerçek TabPFN tahminleri çalışıyor")
                else:
                    print("  ❌ TabPFN tahminleri çalışmıyor")
            else:
                print("  ❌ TabPFN predictor oluşturulamıyor")
        else:
            print("  ❌ TabPFN modülü import edilemiyor")
            
        if tabpfn.get('using_mock') is False:
            print("  🎯 SONUÇ: Gerçek TabPFN/Wright-Fisher analizi kullanılıyor")
        elif tabpfn.get('using_mock') is True:
            print("  ⚠️ SONUÇ: Mock analiz kullanılıyor")
        else:
            print("  ❓ SONUÇ: TabPFN durumu belirsiz")
        
        # Gemini Durumu
        print("\n🤖 GEMİNİ AI DURUMU:")
        gemini = self.results['gemini_status']
        
        if gemini.get('api_key_present'):
            print("  ✅ Gemini API Key mevcut")
        else:
            print("  ❌ Gemini API Key eksik")
            
        if gemini.get('instance_created'):
            print("  ✅ GeminiAI instance oluşturulabiliyor")
        else:
            print("  ❌ GeminiAI instance oluşturulamıyor")
            
        if gemini.get('real_api_call'):
            print("  ✅ Gerçek Gemini API çağrıları çalışıyor")
            print("  🎯 SONUÇ: Gerçek Gemini AI entegrasyonu aktif")
        else:
            print("  ❌ Gemini API çağrıları çalışmıyor")
            print("  ⚠️ SONUÇ: Gemini AI placeholder modunda")
        
        # Öneriler
        print("\n💡 ÖNERİLER:")
        if not tabpfn.get('real_prediction'):
            print("  - TabPFN real prediction'ları aktifleştirin")
        if not gemini.get('real_api_call'):
            print("  - Gemini API bağlantısını kontrol edin")
        if self.results['integration_issues']:
            print("  - Content-Type hatalarını düzeltin")
        if not self.results['simulation_errors']:
            print("  ✅ Kritik hata bulunamadı")
        
        # JSON rapor kaydet
        try:
            with open('debug_analysis_report.json', 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Detaylı debug raporu kaydedildi: debug_analysis_report.json")
        except Exception as e:
            print(f"⚠️ Debug raporu kaydedilemedi: {e}")

def run_debug_analysis():
    """Debug analizi çalıştır"""
    debugger = DebugAnalysis()
    debugger.run_debug_analysis()
    return debugger.results

if __name__ == "__main__":
    print("🐛 Starting NeoMag V7 Debug Analysis...")
    results = run_debug_analysis()
    print("\n✅ Debug analysis completed!") 