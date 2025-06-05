"""
NeoMag V7 - Data Exporters
Simülasyon verilerini farklı formatlarda dışa aktarma fonksiyonları
"""

import csv
import json
import io
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def export_to_csv(bacteria_data, filename=None):
    """
    Bakteri verilerini CSV formatında export et
    
    Args:
        bacteria_data: Bakteri listesi (dict formatında)
        filename: Çıktı dosya adı (opsiyonel)
    
    Returns:
        CSV string veya dosyaya yazma durumu
    """
    try:
        if not bacteria_data:
            logger.warning("No bacteria data to export")
            return None
        
        # CSV için gerekli alanlar
        fieldnames = ['id', 'x', 'y', 'energy', 'fitness', 'generation', 
                     'classification', 'speed', 'size', 'lifetime',
                     'mutation_count', 'food_eaten', 'distance_traveled']
        
        if filename:
            # Dosyaya yaz
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for bacterium in bacteria_data:
                    writer.writerow({k: bacterium.get(k, '') for k in fieldnames})
            logger.info(f"Exported {len(bacteria_data)} bacteria to {filename}")
            return True
        else:
            # String olarak döndür
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for bacterium in bacteria_data:
                writer.writerow({k: bacterium.get(k, '') for k in fieldnames})
            return output.getvalue()
            
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return None


def export_to_json(simulation_data, filename=None, pretty=True):
    """
    Tüm simülasyon verilerini JSON formatında export et
    
    Args:
        simulation_data: Simülasyon verileri (bacteria, stats, history vb.)
        filename: Çıktı dosya adı (opsiyonel)
        pretty: Okunabilir format için indent kullan
    
    Returns:
        JSON string veya dosyaya yazma durumu
    """
    try:
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'version': 'NeoMag V7',
            'data': simulation_data
        }
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, indent=4 if pretty else None)
            logger.info(f"Exported simulation data to {filename}")
            return True
        else:
            return json.dumps(export_data, indent=4 if pretty else None)
            
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return None


def export_to_excel(simulation_data, filename):
    """
    Simülasyon verilerini Excel formatında export et
    
    Args:
        simulation_data: Simülasyon verileri
        filename: Excel dosya adı
    
    Returns:
        Başarı durumu
    """
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Bakteri verileri
            if 'bacteria' in simulation_data:
                df_bacteria = pd.DataFrame(simulation_data['bacteria'])
                df_bacteria.to_excel(writer, sheet_name='Bacteria', index=False)
            
            # İstatistikler
            if 'stats' in simulation_data:
                df_stats = pd.DataFrame([simulation_data['stats']])
                df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Tarihçe
            if 'history' in simulation_data:
                df_history = pd.DataFrame(simulation_data['history'])
                df_history.to_excel(writer, sheet_name='History', index=False)
        
        logger.info(f"Exported simulation data to Excel: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        return False


def export_analysis_report(analysis_data, filename=None):
    """
    AI analiz raporunu markdown formatında export et
    
    Args:
        analysis_data: AI analiz sonuçları
        filename: Rapor dosya adı (opsiyonel)
    
    Returns:
        Markdown string veya dosyaya yazma durumu
    """
    try:
        report = f"""# NeoMag V7 Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Population Overview
{analysis_data.get('summary', 'No summary available')}

## AI Analysis
{analysis_data.get('ai_analysis', 'No AI analysis available')}

## Classifications
"""
        if 'classifications' in analysis_data:
            for cls, count in analysis_data['classifications'].items():
                report += f"- **{cls}**: {count} bacteria\n"
        
        report += f"\n## Performance Metrics\n"
        if 'metrics' in analysis_data:
            for metric, value in analysis_data['metrics'].items():
                report += f"- **{metric}**: {value}\n"
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Exported analysis report to {filename}")
            return True
        else:
            return report
            
    except Exception as e:
        logger.error(f"Error exporting analysis report: {e}")
        return None
