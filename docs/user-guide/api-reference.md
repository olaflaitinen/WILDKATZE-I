# API-Referenz

## Basis-URL

```
https://api.wildkatze.mil/v1
```

## Endpunkte

### POST /v1/analyze

Analysiert Zielgruppenmerkmale.

**Request:**
```json
{
  "demographic_data": {...},
  "behavioral_data": {...}
}
```

**Response:**
```json
{
  "psychographic_profile": {...},
  "confidence": 0.92
}
```

### POST /v1/predict

Vorhersage der Nachrichteneffektivität.

### POST /v1/adapt

Kulturelle Anpassung von Nachrichten.

### POST /v1/counter

Generierung von Gegennarrativen.

### POST /v1/evaluate

Ethische Compliance-Bewertung.

### GET /v1/health

Gesundheitsprüfung.

### GET /v1/metrics

Prometheus-Metriken.
