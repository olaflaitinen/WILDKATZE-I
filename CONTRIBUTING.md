# Beitragsrichtlinien für WILDKATZE-I

## Dokumentenübersicht

Dieses Dokument beschreibt die Richtlinien und Verfahren für Beiträge zum WILDKATZE-I Projekt. Es richtet sich an Entwickler, Forscher und alle Personen, die zur Weiterentwicklung dieses militärischen Sprachmodells beitragen möchten.

Die Einhaltung dieser Richtlinien ist obligatorisch für alle Mitwirkenden. Sie gewährleistet die Qualität, Sicherheit und Konsistenz des Projekts gemäß den Anforderungen des Deutschen Forschungszentrums für Künstliche Intelligenz (DFKI) und der Bundeswehr.

---

## Inhaltsverzeichnis

1. [Verhaltenskodex](#verhaltenskodex)
2. [Zugangsvoraussetzungen](#zugangsvoraussetzungen)
3. [Entwicklungsumgebung](#entwicklungsumgebung)
4. [Branching-Strategie](#branching-strategie)
5. [Commit-Richtlinien](#commit-richtlinien)
6. [Pull-Request-Prozess](#pull-request-prozess)
7. [Code-Review-Prozess](#code-review-prozess)
8. [Code-Qualitätsstandards](#code-qualitätsstandards)
9. [Dokumentationsstandards](#dokumentationsstandards)
10. [Testanforderungen](#testanforderungen)
11. [Sicherheitsrichtlinien](#sicherheitsrichtlinien)
12. [Lizenzierung](#lizenzierung)
13. [Kommunikation](#kommunikation)
14. [Anerkennung von Beiträgen](#anerkennung-von-beiträgen)

---

## Verhaltenskodex

Alle Mitwirkenden verpflichten sich zur Einhaltung des Verhaltenskodex dieses Projekts. Der vollständige Verhaltenskodex ist in der Datei CODE_OF_CONDUCT.md dokumentiert. Die Kernprinzipien umfassen:

### Grundlegende Erwartungen

1. **Professionalität**: Alle Interaktionen erfolgen auf professioneller Ebene entsprechend den Standards einer militärischen Forschungseinrichtung.

2. **Respekt**: Respektvoller Umgang mit allen Projektbeteiligten unabhängig von Herkunft, Geschlecht, Religion oder anderen persönlichen Merkmalen.

3. **Konstruktivität**: Kritik wird konstruktiv und sachbezogen geäußert mit dem Ziel der Verbesserung des Projekts.

4. **Vertraulichkeit**: Vertrauliche Informationen werden entsprechend ihrer Klassifizierung behandelt.

5. **Integrität**: Wissenschaftliche Integrität und Ehrlichkeit in allen Aspekten der Arbeit.

### Unzulässiges Verhalten

Folgendes Verhalten ist strikt untersagt:

- Diskriminierung jeglicher Art
- Belästigung oder Einschüchterung
- Veröffentlichung vertraulicher Informationen
- Absichtliche Sabotage des Projekts
- Missbrauch von Zugriffsrechten
- Umgehung von Sicherheitsmaßnahmen

### Konsequenzen

Verstöße gegen den Verhaltenskodex werden entsprechend der Schwere des Verstoßes geahndet. Mögliche Konsequenzen umfassen:

1. Verwarnung
2. Temporäre Suspendierung der Beitragsrechte
3. Permanenter Ausschluss vom Projekt
4. Meldung an zuständige Behörden bei rechtlichen Verstößen

---

## Zugangsvoraussetzungen

### Allgemeine Anforderungen

Aufgrund der sensiblen Natur dieses Projekts gelten besondere Zugangsvoraussetzungen:

1. **Identitätsverifizierung**: Alle Mitwirkenden müssen ihre Identität verifizieren.

2. **Institutionelle Anbindung**: Mitwirkende müssen einer anerkannten Forschungseinrichtung oder staatlichen Organisation angehören.

3. **Geheimhaltungsvereinbarung**: Unterzeichnung einer Geheimhaltungsvereinbarung (NDA) ist erforderlich.

4. **Sicherheitsüberprüfung**: Je nach Zugriffsebene kann eine Sicherheitsüberprüfung erforderlich sein.

### Zugriffsebenen

| Ebene | Beschreibung | Anforderungen |
|-------|--------------|---------------|
| Observer | Lesezugriff auf öffentliche Dokumentation | Registrierung |
| Contributor | Einreichung von Pull Requests | Identitätsverifizierung, NDA |
| Maintainer | Merge-Rechte, Issue-Management | Institutionelle Anbindung, Erfahrung |
| Core Team | Vollzugriff, Architekturentscheidungen | Sicherheitsüberprüfung, Ernennung |

### Antragsprozess

1. Kontaktaufnahme über die offizielle E-Mail-Adresse
2. Einreichung des Antragsformulars
3. Überprüfung der Qualifikationen
4. Unterzeichnung erforderlicher Vereinbarungen
5. Onboarding und Zugangskonfiguration

---

## Entwicklungsumgebung

### Einrichtung der lokalen Entwicklungsumgebung

Die Einrichtung der Entwicklungsumgebung erfolgt durch Ausführung des Setup-Skripts:

```bash
git clone https://github.com/olaflaitinen/wildkatze-i.git
cd wildkatze-i
chmod +x scripts/setup/setup_environment.sh
./scripts/setup/setup_environment.sh
source .venv/bin/activate
```

### Erforderliche Software

| Software | Mindestversion | Beschreibung |
|----------|----------------|--------------|
| Python | 3.10 | Programmiersprache |
| Git | 2.40 | Versionskontrolle |
| Docker | 24.0 | Containerisierung |
| CUDA | 12.1 | GPU-Beschleunigung |
| PyTorch | 2.1 | Deep Learning Framework |

### IDE-Konfiguration

Es wird empfohlen, Visual Studio Code oder PyCharm mit den folgenden Erweiterungen zu verwenden:

- Python Language Server
- Pylint
- Black Formatter
- isort
- MyPy Type Checker
- GitLens

### Pre-Commit-Hooks

Pre-Commit-Hooks sind obligatorisch und werden automatisch bei der Einrichtung installiert:

```bash
pre-commit install
```

Die Hooks führen folgende Prüfungen durch:

1. Code-Formatierung mit Black
2. Import-Sortierung mit isort
3. Linting mit flake8
4. Typenprüfung mit mypy
5. Sicherheitsanalyse mit bandit

---

## Branching-Strategie

### Haupt-Branches

| Branch | Beschreibung | Schutzstatus |
|--------|--------------|--------------|
| main | Produktionsreifer Code | Geschützt, erfordert Review |
| develop | Integrationsbranche | Geschützt, erfordert CI |

### Feature-Branches

Feature-Branches folgen der Namenskonvention:

```
feature/<issue-nummer>-<kurzbeschreibung>
```

Beispiele:
- `feature/123-attention-optimization`
- `feature/456-cultural-adapter`

### Weitere Branch-Typen

| Typ | Präfix | Beschreibung |
|-----|--------|--------------|
| Bugfix | bugfix/ | Fehlerbehebungen |
| Hotfix | hotfix/ | Dringende Produktionskorrekturen |
| Release | release/ | Veröffentlichungsvorbereitungen |
| Experiment | experiment/ | Experimentelle Entwicklungen |

### Branch-Lebenszyklus

1. **Erstellung**: Branch vom aktuellen `develop` erstellen
2. **Entwicklung**: Änderungen implementieren und committen
3. **Synchronisation**: Regelmäßiges Rebase auf `develop`
4. **Review**: Pull Request erstellen und Review durchführen
5. **Merge**: Nach Genehmigung in `develop` mergen
6. **Löschung**: Branch nach Merge löschen

---

## Commit-Richtlinien

### Conventional Commits

Alle Commits müssen dem Conventional Commits Standard folgen:

```
<typ>(<bereich>): <beschreibung>

[optionaler body]

[optionaler footer]
```

### Commit-Typen

| Typ | Beschreibung | Beispiel |
|-----|--------------|----------|
| feat | Neue Funktionalität | feat(model): add cultural context adapter |
| fix | Fehlerbehebung | fix(attention): correct scaling factor |
| docs | Dokumentationsänderungen | docs(readme): update installation instructions |
| style | Formatierungsänderungen | style(api): apply black formatting |
| refactor | Code-Umstrukturierung | refactor(training): extract optimizer logic |
| test | Testbezogene Änderungen | test(model): add attention unit tests |
| chore | Wartungsarbeiten | chore(deps): update pytorch to 2.2 |
| perf | Leistungsoptimierungen | perf(inference): optimize kv-cache |
| ci | CI/CD-Änderungen | ci(workflow): add model validation |
| security | Sicherheitsverbesserungen | security(api): add input validation |

### Commit-Beschreibung

Die Beschreibung muss:

1. Im Imperativ geschrieben sein ("add" nicht "added")
2. Kleinschreibung am Anfang verwenden
3. Keinen Punkt am Ende haben
4. Maximal 72 Zeichen umfassen

### Commit-Body

Der optionale Body bietet zusätzlichen Kontext:

- Erklärung des "Warum" hinter der Änderung
- Beschreibung komplexer Implementierungsdetails
- Referenzen zu relevanten Issues oder Diskussionen

### Commit-Footer

Der Footer enthält Metadaten:

```
Refs: #123
Co-authored-by: Name <email>
BREAKING CHANGE: description
```

---

## Pull-Request-Prozess

### Erstellung eines Pull Requests

1. **Branch aktualisieren**: Sicherstellen, dass der Branch mit `develop` synchronisiert ist

2. **Selbstprüfung**: Alle Pre-Commit-Hooks erfolgreich durchlaufen

3. **PR erstellen**: Pull Request über GitHub erstellen

4. **Template ausfüllen**: Das PR-Template vollständig ausfüllen

5. **Reviewer zuweisen**: Mindestens zwei Reviewer zuweisen

6. **Labels hinzufügen**: Entsprechende Labels vergeben

### PR-Template-Anforderungen

Jeder Pull Request muss folgende Informationen enthalten:

1. **Beschreibung**: Klare Beschreibung der Änderungen
2. **Motivation**: Begründung für die Änderungen
3. **Art der Änderung**: Klassifizierung (Feature, Bugfix, etc.)
4. **Checkliste**: Bestätigung aller Qualitätskriterien
5. **Testnachweis**: Beschreibung der durchgeführten Tests
6. **Dokumentation**: Bestätigung der Dokumentationsaktualisierung

### Größenbeschränkungen

| Kategorie | Maximale LOC | Review-Anforderung |
|-----------|--------------|-------------------|
| Klein | 50 | 1 Reviewer |
| Mittel | 200 | 2 Reviewer |
| Groß | 500 | 2 Reviewer + Architect |
| Sehr groß | 500+ | Aufteilen empfohlen |

### Merge-Anforderungen

Ein Pull Request kann gemerged werden, wenn:

1. Alle CI-Checks erfolgreich sind
2. Mindestens zwei Approvals vorliegen
3. Alle Kommentare bearbeitet wurden
4. Keine offenen Change Requests existieren
5. Der Branch mit der Zielbranch synchronisiert ist

---

## Code-Review-Prozess

### Reviewer-Verantwortlichkeiten

Reviewer sind verantwortlich für:

1. **Korrektheit**: Überprüfung der funktionalen Korrektheit
2. **Qualität**: Bewertung der Code-Qualität
3. **Sicherheit**: Identifizierung von Sicherheitsproblemen
4. **Architektur**: Prüfung der architektonischen Konformität
5. **Dokumentation**: Verifizierung der Dokumentation
6. **Tests**: Bewertung der Testabdeckung

### Review-Kommentare

Kommentare sollten:

- Konstruktiv und sachlich sein
- Konkrete Verbesserungsvorschläge enthalten
- Mit dem entsprechenden Schweregrad gekennzeichnet sein

### Schweregrade

| Grad | Symbol | Beschreibung |
|------|--------|--------------|
| Blocking | MUST | Muss vor Merge behoben werden |
| Important | SHOULD | Sollte behoben werden |
| Suggestion | COULD | Verbesserungsvorschlag |
| Question | ASK | Frage zur Klärung |
| Praise | NICE | Positives Feedback |

### Zeitrahmen

| Kategorie | Erstreaktion | Vollständiges Review |
|-----------|--------------|---------------------|
| Kritisch | 4 Stunden | 24 Stunden |
| Normal | 24 Stunden | 72 Stunden |
| Niedrig | 72 Stunden | 1 Woche |

---

## Code-Qualitätsstandards

### Python-Stilrichtlinien

Der Code muss PEP 8 entsprechen mit folgenden Erweiterungen:

1. **Zeilenlänge**: Maximal 100 Zeichen
2. **Einrückung**: 4 Leerzeichen
3. **Strings**: Doppelte Anführungszeichen bevorzugt
4. **Imports**: Gruppiert und sortiert mit isort

### Type Hints

Type Hints sind obligatorisch für alle öffentlichen Funktionen:

```python
def analyze_cultural_context(
    text: str,
    target_culture: str,
    confidence_threshold: float = 0.8
) -> Tuple[float, List[str]]:
    """Analyze cultural appropriateness of text."""
    ...
```

### Docstrings

Alle Module, Klassen und öffentlichen Funktionen erfordern Google-Style Docstrings:

```python
def predict_resonance(
    message: str,
    audience: AudienceProfile,
    culture: str
) -> ResonancePrediction:
    """Predict message resonance for target audience.
    
    Analyzes the expected effectiveness of a message for
    a specific target audience within a cultural context.
    
    Args:
        message: The message content to analyze.
        audience: Profile of the target audience.
        culture: ISO culture code (e.g., "de-DE").
        
    Returns:
        ResonancePrediction containing score and recommendations.
        
    Raises:
        ValidationError: If message is empty or culture invalid.
        CulturalDataError: If culture data is unavailable.
        
    Example:
        >>> result = predict_resonance(
        ...     "Safety for our families",
        ...     moderate_audience,
        ...     "ar-SA"
        ... )
        >>> print(result.score)
        0.82
    """
    ...
```

### Komplexitätsmetriken

| Metrik | Maximum | Empfohlen |
|--------|---------|-----------|
| Zyklomatische Komplexität | 15 | 10 |
| Kognitive Komplexität | 25 | 15 |
| Funktionslänge | 50 LOC | 30 LOC |
| Klassenlänge | 500 LOC | 300 LOC |
| Parameterzahl | 8 | 5 |

### Qualitätstools

| Tool | Zweck | Mindest-Score |
|------|-------|---------------|
| pylint | Statische Analyse | 9.0/10 |
| flake8 | Style-Checking | Keine Fehler |
| mypy | Type-Checking | Keine Fehler |
| bandit | Sicherheitsanalyse | Keine High/Critical |
| black | Formatierung | Konform |

---

## Dokumentationsstandards

### Anforderungen

Jede Codeänderung muss entsprechende Dokumentation umfassen:

1. **Inline-Kommentare**: Für komplexe Logik
2. **Docstrings**: Für alle öffentlichen APIs
3. **README-Updates**: Bei neuen Features
4. **API-Dokumentation**: Bei API-Änderungen

### Sprache

- **Code-Kommentare**: Englisch
- **Benutzer-Dokumentation**: Deutsch
- **API-Dokumentation**: Englisch

### Formatierung

Dokumentation folgt diesen Richtlinien:

1. Keine Emojis
2. Keine Emdashes (Gedankenstriche)
3. Formeller, akademischer Stil
4. Passive Konstruktionen wo angemessen
5. Technische Präzision

---

## Testanforderungen

### Testabdeckung

| Komponente | Mindestabdeckung | Zielabdeckung |
|------------|-----------------|---------------|
| Model | 80% | 90% |
| Training | 75% | 85% |
| Inference | 85% | 95% |
| API | 90% | 95% |
| Utils | 90% | 95% |

### Testtypen

1. **Unit-Tests**: Isolierte Komponentenprüfung
2. **Integrationstests**: Komponenteninteraktion
3. **Performance-Tests**: Latenz- und Durchsatzmessung
4. **Compliance-Tests**: Ethische und rechtliche Konformität

### Testframeworks

| Framework | Verwendung |
|-----------|------------|
| pytest | Unit- und Integrationstests |
| pytest-cov | Abdeckungsmessung |
| pytest-benchmark | Performance-Benchmarks |
| hypothesis | Property-Based Testing |

---

## Sicherheitsrichtlinien

### Verbotene Praktiken

1. Hardcodierte Zugangsdaten
2. Unsichere Deserialisierung
3. SQL/Command Injection Risiken
4. Unvalidierte Benutzereingaben
5. Logging sensitiver Daten

### Erforderliche Praktiken

1. Eingabevalidierung für alle externen Daten
2. Sichere Zufallszahlengenerierung
3. Verschlüsselung sensibler Daten
4. Least Privilege Prinzip
5. Regelmäßige Abhängigkeitsupdates

### Sicherheitsüberprüfung

Alle Pull Requests durchlaufen automatisch:

1. Bandit Sicherheitsanalyse
2. Snyk Abhängigkeitsprüfung
3. Trivy Container-Scanning
4. TruffleHog Secret-Detection

---

## Lizenzierung

### EUPL v1.2

Alle Beiträge werden unter der European Union Public Licence (EUPL) Version 1.2 lizenziert. Durch die Einreichung eines Beitrags stimmen Sie der Lizenzierung unter EUPL v1.2 zu.

### Contributor License Agreement

Vor dem ersten Beitrag ist die Unterzeichnung eines Contributor License Agreement (CLA) erforderlich. Das CLA bestätigt:

1. Berechtigung zur Einreichung des Beitrags
2. Originalität des Beitrags
3. Zustimmung zur Lizenzierung unter EUPL v1.2
4. Abtretung erforderlicher Rechte an das Projekt

---

## Kommunikation

### Kommunikationskanäle

| Kanal | Verwendung | Reaktionszeit |
|-------|------------|---------------|
| GitHub Issues | Bugs, Features | 48 Stunden |
| GitHub Discussions | Allgemeine Fragen | 72 Stunden |
| E-Mail | Vertrauliche Anfragen | 24 Stunden |
| Meetings | Architekturentscheidungen | Nach Vereinbarung |

### Issue-Erstellung

Issues sollten folgende Informationen enthalten:

1. Klare Beschreibung des Problems oder Vorschlags
2. Reproduktionsschritte bei Bugs
3. Erwartetes Verhalten
4. Tatsächliches Verhalten
5. Umgebungsinformationen
6. Relevante Logs oder Screenshots

---

## Anerkennung von Beiträgen

### Beitragstypen

Alle Beiträge werden anerkannt, einschließlich:

- Code-Beiträge
- Dokumentation
- Bug-Reports
- Feature-Vorschläge
- Code-Review
- Community-Support

### Anerkennung

Mitwirkende werden anerkannt durch:

1. Eintrag in CONTRIBUTORS.md
2. Git-Historie
3. Release Notes
4. Wissenschaftliche Publikationen (bei signifikanten Beiträgen)

---

## Kontakt

Bei Fragen zu diesen Richtlinien:

- **E-Mail**: contributing@wildkatze.mil
- **Projektleitung**: Olaf Laitinen

---

Copyright 2026 olaflaitinen. Alle Rechte vorbehalten.
Dieses Dokument ist Teil der WILDKATZE-I Dokumentation und unterliegt der EUPL v1.2 Lizenz.
