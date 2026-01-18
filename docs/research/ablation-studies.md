# Ablationsstudien

## Architektur-Ablationen

### Attention Heads

| Heads | Resonanz | Speicher |
|-------|----------|----------|
| 32 | 75.1% | 48 GB |
| 64 | 79.2% | 56 GB |
| 128 | 79.8% | 72 GB |

### Layer-Anzahl

| Layers | Resonanz | Training Time |
|--------|----------|---------------|
| 24 | 71.2% | 48h |
| 48 | 79.2% | 96h |
| 80 | 80.1% | 168h |

## Cultural Context Dimension

| Dimension | Kulturelle Angemessenheit |
|-----------|---------------------------|
| 512 | 7.8/10 |
| 1024 | 8.3/10 |
| 2048 | 8.4/10 |

## Schlussfolgerung

Die gewählte Konfiguration (48 Layers, 64 Heads, 1024 Cultural Dim) bietet das beste Verhältnis von Leistung zu Ressourcenverbrauch.
