import keras

from models.train import train

class TestTrain:
    def test_load_model(self):
        model, history = train()
        assert isinstance(model, keras.Model)
    
