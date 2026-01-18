# Schnellstart

## Voraussetzungen

Stellen Sie sicher, dass WILDKATZE-I installiert ist (siehe [Installation](installation.md)).

## Grundlegende Verwendung

### 1. Modell laden

```python
from wildkatze import WildkatzeConfig, WildkatzeForCausalLM

config = WildkatzeConfig()
model = WildkatzeForCausalLM(config)
```

### 2. Kulturelle Sensitivitätsanalyse

```python
from wildkatze.evaluation import CulturalSensitivityAnalyzer

analyzer = CulturalSensitivityAnalyzer("data/samples/cultural_contexts.json")
score, issues = analyzer.evaluate(
    "Gemeinsam bauen wir eine sichere Zukunft.",
    target_culture="de-DE"
)
print(f"Score: {score}/10")
```

### 3. API-Server starten

```bash
uvicorn wildkatze.deployment.api_server:app --host 0.0.0.0 --port 8080
```

### 4. API-Anfrage

```bash
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"message_content": "Test", "target_audience": "general", "culture": "de-DE"}'
```
