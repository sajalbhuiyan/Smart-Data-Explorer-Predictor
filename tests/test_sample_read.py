import pandas as pd

def test_read_sample():
    df = pd.read_csv('sample_data.csv')
    assert df.shape[0] > 0
