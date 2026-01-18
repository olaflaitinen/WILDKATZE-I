# DSGVO-Compliance

## Datenschutzprinzipien

WILDKATZE-I implementiert die folgenden DSGVO-Anforderungen:

### Datenminimierung

- Nur notwendige Daten werden verarbeitet
- Automatische Löschung nach Retention Period

### Zweckbindung

- Daten werden nur für definierte Zwecke verwendet
- Keine Weitergabe an Dritte ohne Zustimmung

### Transparenz

- Klare Dokumentation der Datenverarbeitung
- Audit-Protokolle

## Technische Maßnahmen

1. **PII-Redaktion**: Automatische Entfernung personenbezogener Daten aus Logs
2. **Verschlüsselung**: AES-256 at Rest, TLS 1.3 in Transit
3. **Zugriffskontrolle**: RBAC mit minimalen Berechtigungen
4. **Audit-Logging**: Vollständige Protokollierung aller Zugriffe

## Betroffenenrechte

- Auskunftsrecht
- Recht auf Löschung
- Recht auf Berichtigung
- Recht auf Datenportabilität
