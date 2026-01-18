# Beispiele

## Kulturelle Sensitivitätsanalyse

```python
from wildkatze.evaluation import CulturalSensitivityAnalyzer

analyzer = CulturalSensitivityAnalyzer("data/samples/cultural_contexts.json")
score, issues = analyzer.evaluate(
    "Gemeinsam für eine sichere Zukunft.",
    target_culture="de-DE"
)
print(f"Kulturelle Angemessenheit: {score}/10")
```

## Bias-Erkennung

```python
from wildkatze.evaluation import BiasDetector

detector = BiasDetector()
report = detector.get_report("Beispieltext zur Analyse")
print(f"Bias-Score: {report['score']}")
```

## API-Nutzung

```bash
curl -X POST https://api.wildkatze.mil/v1/predict \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"message_content": "Test", "target_audience": "general", "culture": "ar-SA"}'
```
