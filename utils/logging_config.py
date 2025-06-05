"""
NeoMag V7 - Logging Configuration
Merkezi loglama yapılandırması
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import colorlog


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    console_output: bool = True,
    colorize: bool = True,
    log_format: str = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Merkezi loglama yapılandırması
    
    Args:
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log dosyası yolu (None ise otomatik oluştur)
        console_output: Konsola çıktı ver
        colorize: Konsol çıktısını renklendir
        log_format: Özel log formatı (None ise varsayılan)
        max_file_size: Maksimum log dosya boyutu (bytes)
        backup_count: Saklanacak yedek log dosya sayısı
    """
    # Log dizinini oluştur
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log dosya adı
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"neomag_v7_{timestamp}.log"
    
    # Root logger'ı yapılandır
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Mevcut handler'ları temizle
    root_logger.handlers = []
    
    # Log formatı
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    
    # Dosya handler'ı
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Konsol handler'ı
    if console_output:
        console_handler = logging.StreamHandler()
        
        if colorize:
            # Renkli konsol formatı
            color_format = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s"
            console_formatter = colorlog.ColoredFormatter(
                color_format,
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            console_formatter = logging.Formatter(log_format)
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # İlk log mesajı
    logging.info(f"Logging configured - Level: {log_level}, File: {log_file}")


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Modül için özel logger oluştur
    
    Args:
        module_name: Modül adı
    
    Returns:
        Logger instance
    """
    return logging.getLogger(module_name)


def configure_external_loggers():
    """
    Harici kütüphanelerin log seviyelerini ayarla
    """
    # Gürültülü kütüphaneleri sustur
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('engineio').setLevel(logging.WARNING)
    logging.getLogger('socketio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    # Önemli kütüphaneleri etkinleştir
    logging.getLogger('neomag_v7').setLevel(logging.DEBUG)
    logging.getLogger('tabpfn').setLevel(logging.INFO)


def log_exception(logger: logging.Logger, exception: Exception, context: str = ""):
    """
    Exception'ı detaylı şekilde logla
    
    Args:
        logger: Logger instance
        exception: Yakalanan exception
        context: Ek bağlam bilgisi
    """
    import traceback
    
    error_msg = f"Exception occurred"
    if context:
        error_msg += f" in {context}"
    
    logger.error(f"{error_msg}: {type(exception).__name__}: {str(exception)}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")


def create_performance_logger(name: str = "performance") -> logging.Logger:
    """
    Performans ölçümleri için özel logger
    
    Args:
        name: Logger adı
    
    Returns:
        Performance logger
    """
    perf_logger = logging.getLogger(f"neomag_v7.{name}")
    
    # Performans log dosyası
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    
    perf_format = "%(asctime)s - %(message)s"
    perf_formatter = logging.Formatter(perf_format)
    perf_handler.setFormatter(perf_formatter)
    
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    return perf_logger


def log_system_info():
    """
    Sistem bilgilerini logla
    """
    import platform
    import sys
    
    logger = logging.getLogger(__name__)
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not installed")
    
    logger.info("========================")


# Varsayılan yapılandırma
def init_default_logging():
    """
    Varsayılan logging yapılandırmasını başlat
    """
    setup_logging(
        log_level="INFO",
        console_output=True,
        colorize=True
    )
    configure_external_loggers()
    log_system_info()


# Modül import edildiğinde otomatik başlat
if __name__ != "__main__":
    init_default_logging()
