import tempfile
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd


def test_model_discovery_two_stage(monkeypatch):
    """Test that FutureDownscaler discovers two-stage precipitation models."""
    import src.inference.downscale_future as df

    # Create temporary models directory with expected filenames
    tmp = tempfile.TemporaryDirectory()
    models_dir = Path(tmp.name) / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    # create placeholder files
    (models_dir / 'xgboost_tas.pkl').write_text('dummy')
    (models_dir / 'xgboost_pr_occ.pkl').write_text('dummy')
    (models_dir / 'xgboost_pr_amt.pkl').write_text('dummy')

    # Monkeypatch DownscalingModel.load to a lightweight dummy
    class DummyModel:
        def __init__(self):
            self.algorithm = 'xgboost'
            self.training_history = {}
        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(df, 'DownscalingModel', df.DownscalingModel)
    monkeypatch.setattr(df.DownscalingModel, 'load', staticmethod(lambda p: DummyModel()))

    # Monkeypatch ClimateDataPreprocessor to return a tiny CRU dataset
    class DummyPreprocessor:
        def __init__(self, base_path=None):
            pass
        def load_cru(self):
            times = pd.date_range('1980-01-01', periods=1, freq='MS')
            ds = xr.Dataset(
                {'tas': (('time', 'lat', 'lon'), np.zeros((1, 1, 1)))},
                coords={'time': times, 'lat': [0.0], 'lon': [0.0]}
            )
            return ds, {}

    monkeypatch.setattr(df, 'ClimateDataPreprocessor', DummyPreprocessor)

    # Instantiate downscaler (should not raise)
    downscaler = df.FutureDownscaler(models_path=str(models_dir), base_data_path=str(models_dir))

    assert getattr(downscaler, 'precip_two_stage', False) is True
    assert downscaler.temp_model is not None
