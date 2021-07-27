# Meuseum Model Notes

### Train Model

```sh
python3 train2.py ../data/41586_2020_3052_MOESM8_ESM.txt ../data/41586_2020_3052_MOESM9_ESM.txt parameter1.txt
```

## Model Evaluatation

Run model on a library. Save predictions and show performance metrics.

```sh
python3 analyse.py parameter1.txt
```

### Prediction Metrics

| Test Library | R2 Score | Pearson's Correlation | Spearman's Correlation |
| TL (trained) | 0.92 | 0.96 | 0.95 |
| CNL | 0.77 | | |
| RL | 0.81 | | |
| CHRVL | 0.57 | 0.76 | 0.74 |
