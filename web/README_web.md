# 🧬 NeoMag V7 Web Interface

**Modern ve İnteraktif Bilimsel Simülasyon Arayüzü**

## 🚀 Özellikler

### 🎮 Real-Time Kontrol
- **Start/Pause/Stop** simülasyon kontrolü
- **Live FPS** göstergesi ve performans metrikleri  
- **Dinamik bakteri ekleme** (özelleştirilebilir sayıda)
- **Anlık veri görselleştirmesi**

### 📊 Bilimsel Veri Görselleştirmesi
- **Real-time grafikler** (Plotly.js ile)
- **Popülasyon istatistikleri** 
- **Genetik çeşitlilik metrikleri**
- **AI performans göstergeleri**

### 🎨 Modern Arayüz
- **Responsive tasarım** (mobil uyumlu)
- **Dark theme** ile profesyonel görünüm
- **Smooth animasyonlar** ve geçişler
- **Notification sistemi** ile kullanıcı bildirimleri

### 🔬 Bilimsel Canvas
- **2D bakteri görselleştirmesi** (fitness tabanlı renklendirme)
- **Besin parçacıkları** glow efektleri ile
- **Grid sistemi** ve koordinat gösterimi
- **Screenshot alma** özelliği

## 🛠️ Kurulum ve Çalıştırma

### 1. Gereksinimler
```bash
pip install flask flask-socketio
```

### 2. Sunucuyu Başlatma
Socket.IO problemleri yaşarsanız `simple_server.py` dosyasını kullanabilirsiniz.
```bash
cd neomag_v7/web
python simple_server.py
```

### 3. Web Arayüzüne Erişim
```
http://localhost:5000
```

## 📱 Arayüz Bileşenleri

### Sol Panel - Kontroller
- **🎮 Simulation Control**: Start, pause, stop butonları
- **🦠 Population Control**: Bakteri sayısı ayarları
- **📊 Data Export**: Bilimsel veri dışa aktarma

### Merkez Panel - Simülasyon Canvas
- **Real-time görselleştirme**: Bakteriler ve besin parçacıkları
- **Live overlay**: Anlık simülasyon verileri
- **Interactive canvas**: Screenshot alma

### Sağ Panel - Data Dashboard
- **📈 Population Statistics**: Canlı popülasyon metrikleri
- **🧬 Genetic Metrics**: Genetik çeşitlilik verileri  
- **📊 Real-time Charts**: Fitness ve popülasyon grafikleri

## 🎯 Kullanım Adımları

### 1. Simülasyon Başlatma
1. Sol panelden **bakteri sayısını** ayarlayın (10-500)
2. **🚀 Start Simulation** butonuna tıklayın
3. Bağlantı durumunu header'da kontrol edin

### 2. Simülasyon İzleme
- Canvas'ta bakterilerin **hareketini** izleyin
- Sağ panelde **live metrikleri** takip edin
- Grafiklerde **trend'leri** gözlemleyin

### 3. Veri Dışa Aktarma
- **💾 Export Scientific Data**: JSON formatında tam veri
- **📸 Screenshot**: Canvas'ın PNG görüntüsü

## 🔧 Teknik Detaylar

### Frontend Teknolojileri
- **HTML5 Canvas**: 2D simülasyon görselleştirmesi
- **Socket.IO**: Real-time veri iletişimi
- **Plotly.js**: İnteraktif grafik görselleştirmesi
- **CSS Grid & Flexbox**: Responsive layout

### Backend API Endpoints
```python
POST /api/start_simulation     # Simülasyonu başlat
POST /api/stop_simulation      # Simülasyonu durdur  
POST /api/pause_resume         # Pause/resume toggle
POST /api/add_bacteria         # Bakteri ekle
GET  /api/simulation_data      # Anlık simülasyon verisi
GET  /api/scientific_export    # Bilimsel veri dışa aktarma
```

### WebSocket Events
```javascript
'connect'           // Sunucuya bağlantı
'disconnect'        // Bağlantı kopması
'simulation_update' // Real-time simülasyon verisi
```

## 🎨 Görsel Öğeler

### Bakteri Görselleştirmesi
- **Renk**: Fitness değerine göre (kırmızı=düşük, yeşil=yüksek)
- **Boyut**: Bakteri büyüklüğü
- **Energy Ring**: Enerji seviyesi göstergesi
- **Generation Label**: Jenerasyon numarası

### Besin Parçacıkları
- **Turuncu renk** ile belirgin görünüm
- **Glow efekti** dikkat çekmek için
- **Boyut**: Enerji içeriğine göre

### UI Renk Paleti
```css
--primary-color: #2E7D32    /* Yeşil - Ana tema */
--secondary-color: #1976D2  /* Mavi - İkincil */
--accent-color: #FF6F00     /* Turuncu - Vurgu */
--success-color: #4CAF50    /* Başarı */
--warning-color: #FF9800    /* Uyarı */
--error-color: #F44336      /* Hata */
```

## 📊 Performance Metrikleri

### Real-time Göstergeler
- **FPS**: Frames per second
- **Step Counter**: Simülasyon adımı
- **Population Count**: Bakteri sayısı
- **Food Count**: Besin parçacık sayısı

### Bilimsel Metrikler
- **Average Fitness**: Ortalama fitness değeri
- **Average Age**: Ortalama yaş
- **Average Energy**: Ortalama enerji seviyesi
- **Generation**: Ortalama jenerasyon

## 🔍 Troubleshooting

### Yaygın Sorunlar

**Bağlantı Problemi**
```
Solution: Server'ın çalıştığından emin olun
Check: http://localhost:5000 erişimi
```

**Simülasyon Başlamıyor**
```
Solution: NeoMag V7 modüllerinin yüklü olduğunu kontrol edin
Check: Backend log'larını inceleyin
```

**Canvas Görünmüyor**
```
Solution: Tarayıcı uyumluluğunu kontrol edin
Check: JavaScript konsol hatalarını inceleyin
```

### Debug Modunda Çalıştırma
```python
# simple_server.py içinde
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 🌐 Browser Uyumluluğu

### Desteklenen Tarayıcılar
- ✅ **Chrome 90+**
- ✅ **Firefox 88+**  
- ✅ **Safari 14+**
- ✅ **Edge 90+**

### Gerekli Özellikler
- Canvas 2D API
- WebSocket desteği
- ES6+ JavaScript
- CSS Grid & Flexbox

## 🚀 Gelecek Özellikler

### v7.1 Planları
- [ ] **3D görselleştirme** (Three.js)
- [ ] **VR/AR desteği**
- [ ] **Multi-user simülasyonları**
- [ ] **AI model karşılaştırması**

### v7.2 Planları  
- [ ] **Real-time collaboration**
- [ ] **Cloud simülasyonları**
- [ ] **Mobile app** (React Native)
- [ ] **API key sistemi**

## 📞 Destek

**Web arayüzü ile ilgili sorunlar için:**
- GitHub Issues: `/neomag-v7/issues`
- Email: `web-support@neomag.dev`
- Documentation: `/docs/web-interface`

---

**NeoMag V7 Web Interface - Where Science Meets Modern UI** 🧬✨ 
