"""
NeoMag V7 - Visualization Utilities
Simülasyon görselleştirme yardımcı fonksiyonları
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Stil ayarları
plt.style.use('dark_background')
sns.set_palette("viridis")


def create_population_chart(bacteria_data: List[Dict]) -> go.Figure:
    """
    Bakteri popülasyonu dağılım grafiği
    
    Args:
        bacteria_data: Bakteri verileri listesi
    
    Returns:
        Plotly figure
    """
    if not bacteria_data:
        return go.Figure()
    
    # DataFrame'e dönüştür
    df = pd.DataFrame(bacteria_data)
    
    # Sınıflandırma dağılımı
    class_counts = df['classification'].value_counts()
    
    colors = {
        'elite': '#ffd700',
        'veteran': '#4169e1', 
        'strong': '#32cd32',
        'energetic': '#ff6347',
        'young': '#00bfff',
        'basic': '#ff8c00'
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=class_counts.index,
            y=class_counts.values,
            marker_color=[colors.get(c, '#ffffff') for c in class_counts.index],
            text=class_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Bakteri Popülasyon Dağılımı",
        xaxis_title="Sınıf",
        yaxis_title="Sayı",
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def create_fitness_distribution(bacteria_data: List[Dict]) -> go.Figure:
    """
    Fitness dağılım histogramı
    
    Args:
        bacteria_data: Bakteri verileri
    
    Returns:
        Plotly figure
    """
    if not bacteria_data:
        return go.Figure()
    
    fitness_values = [b['fitness'] for b in bacteria_data]
    
    fig = go.Figure(data=[
        go.Histogram(
            x=fitness_values,
            nbinsx=30,
            marker_color='#3b82f6',
            marker_line_color='white',
            marker_line_width=1
        )
    ])
    
    fig.update_layout(
        title="Fitness Dağılımı",
        xaxis_title="Fitness",
        yaxis_title="Frekans",
        template="plotly_dark",
        bargap=0.1
    )
    
    # Ortalama çizgisi
    avg_fitness = np.mean(fitness_values)
    fig.add_vline(
        x=avg_fitness,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Ortalama: {avg_fitness:.3f}"
    )
    
    return fig


def create_energy_scatter(bacteria_data: List[Dict]) -> go.Figure:
    """
    Enerji vs Fitness scatter plot
    
    Args:
        bacteria_data: Bakteri verileri
    
    Returns:
        Plotly figure
    """
    if not bacteria_data:
        return go.Figure()
    
    df = pd.DataFrame(bacteria_data)
    
    fig = px.scatter(
        df,
        x='energy',
        y='fitness',
        color='classification',
        size='generation',
        hover_data=['id', 'lifetime'],
        color_discrete_map={
            'elite': '#ffd700',
            'veteran': '#4169e1',
            'strong': '#32cd32',
            'energetic': '#ff6347',
            'young': '#00bfff',
            'basic': '#ff8c00'
        },
        template="plotly_dark"
    )
    
    fig.update_layout(
        title="Enerji vs Fitness İlişkisi",
        xaxis_title="Enerji",
        yaxis_title="Fitness"
    )
    
    return fig


def create_generation_timeline(history_data: List[Dict]) -> go.Figure:
    """
    Jenerasyon gelişim zaman serisi
    
    Args:
        history_data: Tarihçe verileri
    
    Returns:
        Plotly figure
    """
    if not history_data:
        return go.Figure()
    
    df = pd.DataFrame(history_data)
    
    fig = go.Figure()
    
    # Çoklu metrikler
    metrics = ['avg_fitness', 'avg_energy', 'total_bacteria']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    for metric, color in zip(metrics, colors):
        if metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines+markers',
                name=metric.replace('_', ' ').title(),
                line=dict(color=color, width=2),
                marker=dict(size=4)
            ))
    
    fig.update_layout(
        title="Simülasyon Gelişimi",
        xaxis_title="Adım",
        yaxis_title="Değer",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def create_heatmap(bacteria_positions: List[Tuple[float, float]], 
                  canvas_width: int = 1200, 
                  canvas_height: int = 600) -> go.Figure:
    """
    Bakteri yoğunluk haritası
    
    Args:
        bacteria_positions: (x, y) pozisyon listesi
        canvas_width: Canvas genişliği
        canvas_height: Canvas yüksekliği
    
    Returns:
        Plotly figure
    """
    if not bacteria_positions:
        return go.Figure()
    
    x_coords = [pos[0] for pos in bacteria_positions]
    y_coords = [pos[1] for pos in bacteria_positions]
    
    fig = go.Figure(data=go.Histogram2d(
        x=x_coords,
        y=y_coords,
        colorscale='Viridis',
        nbinsx=30,
        nbinsy=20,
        showscale=True
    ))
    
    fig.update_layout(
        title="Bakteri Yoğunluk Haritası",
        xaxis_title="X Koordinat",
        yaxis_title="Y Koordinat",
        template="plotly_dark",
        xaxis=dict(range=[0, canvas_width]),
        yaxis=dict(range=[0, canvas_height])
    )
    
    return fig


def create_performance_metrics(perf_data: Dict) -> go.Figure:
    """
    Performans metrikleri gösterge paneli
    
    Args:
        perf_data: Performans verileri
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # FPS göstergesi
    if 'fps' in perf_data:
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=perf_data['fps'],
            title={'text': "FPS"},
            delta={'reference': 60},
            gauge={
                'axis': {'range': [None, 120]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 120], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            },
            domain={'x': [0, 0.5], 'y': [0, 1]}
        ))
    
    # CPU/GPU kullanımı
    if 'cpu_usage' in perf_data:
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=perf_data['cpu_usage'],
            title={'text': "CPU %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            },
            domain={'x': [0.5, 1], 'y': [0, 1]}
        ))
    
    fig.update_layout(
        title="Performans Metrikleri",
        template="plotly_dark",
        height=300
    )
    
    return fig


def save_plot(fig: go.Figure, filename: str, format: str = 'png'):
    """
    Grafiği dosyaya kaydet
    
    Args:
        fig: Plotly figure
        filename: Dosya adı
        format: Dosya formatı (png, html, svg)
    """
    try:
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'svg':
            fig.write_image(filename, format='svg')
        else:
            logger.error(f"Unsupported format: {format}")
            return
        
        logger.info(f"Plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving plot: {e}")


def create_summary_dashboard(simulation_data: Dict) -> Dict[str, go.Figure]:
    """
    Tüm grafikleri içeren özet dashboard
    
    Args:
        simulation_data: Simülasyon verileri
    
    Returns:
        Figure dictionary
    """
    figures = {}
    
    if 'bacteria' in simulation_data:
        bacteria_data = simulation_data['bacteria']
        figures['population'] = create_population_chart(bacteria_data)
        figures['fitness'] = create_fitness_distribution(bacteria_data)
        figures['energy_scatter'] = create_energy_scatter(bacteria_data)
        
        positions = [(b['x'], b['y']) for b in bacteria_data]
        figures['heatmap'] = create_heatmap(positions)
    
    if 'history' in simulation_data:
        figures['timeline'] = create_generation_timeline(simulation_data['history'])
    
    if 'performance' in simulation_data:
        figures['performance'] = create_performance_metrics(simulation_data['performance'])
    
    return figures
