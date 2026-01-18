"""
module utils.py
"""
import pytest
import pandas as pd
from pathlib import Path
from src.utils import DataLoader, ConfigManager, Logger

class TestDataLoader:
    """Test for DataLoader"""
    
    def test_load_csv(self, tmp_path):
        """Test Load CSV"""
        # Crea file CSV temporaneo
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4")
        
        loader = DataLoader()
        df = loader.load_csv(str(csv_file))
        
        assert len(df) == 2
        assert list(df.columns) == ['col1', 'col2']
    
    def test_save_parquet(self, tmp_path):
        """Test save Parquet"""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        
        loader = DataLoader()
        output_file = tmp_path / "test.parquet"
        loader.save_parquet(df, str(output_file))
        
        # Verifica che il file esista
        assert output_file.exists()

class TestConfigManager:
    """Test per ConfigManager"""
    
    def test_load_json_config(self, tmp_path):
        """Test caricamento config JSON"""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"key": "value", "number": 42}')
        
        manager = ConfigManager()
        config = manager.load_json(str(config_file))
        
        assert config['key'] == 'value'
        assert config['number'] == 42
    
    def test_get_database_config(self):
        """Test configurazione database"""
        from src.utils import DatabaseConfig
        
        config = DatabaseConfig()
        db_config = config.get_config('postgres')
        
        assert 'host' in db_config
        assert 'port' in db_config

class TestLogger:
    """Test Logger"""
    
    def test_logger_creation(self, tmp_path):
        """Test creazione logger"""
        log_file = tmp_path / "test.log"
        
        logger = Logger.get_logger("test_logger", str(log_file))
        
        # Test logging
        logger.info("Test message")
        
        # Verifica che il file di log esista e contenga il messaggio
        assert log_file.exists()
        assert "Test message" in log_file.read_text()

class TestDataQualityUtils:
    """Test utility quality dates"""
    
    def test_calculate_data_quality_metrics(self):
        """Test metriche qualità dati"""
        from src.utils import DataQualityChecker
        
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['a', 'b', 'c', None],
            'C': [1.0, 2.0, 3.0, 4.0]
        })
        
        checker = DataQualityChecker()
        metrics = checker.calculate_metrics(df)
        
        assert 'completeness' in metrics
        assert 'uniqueness' in metrics
        assert metrics['row_count'] == 4

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
