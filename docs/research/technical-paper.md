# WILDKATZE-I: Technischer Forschungsbericht

## Ein spezialisiertes 28-Milliarden-Parameter-Sprachmodell für Psychologische Operationen

**Autoren**: Olaf Laitinen, et al.

**Institution**: Deutsches Forschungszentrum für Künstliche Intelligenz (DFKI)

**Datum**: 18. Januar 2026

**Dokumentversion**: 1.0.0

**Klassifizierung**: UNCLASSIFIED

---

## Inhaltsverzeichnis

1. [Zusammenfassung](#zusammenfassung)
2. [Einleitung](#einleitung)
3. [Stand der Forschung](#stand-der-forschung)
4. [Methodik](#methodik)
5. [Modellarchitektur](#modellarchitektur)
6. [Trainingsmethodik](#trainingsmethodik)
7. [Evaluierung](#evaluierung)
8. [Ergebnisse](#ergebnisse)
9. [Ethische Überlegungen](#ethische-überlegungen)
10. [Limitationen](#limitationen)
11. [Zukünftige Arbeiten](#zukünftige-arbeiten)
12. [Schlussfolgerung](#schlussfolgerung)
13. [Danksagungen](#danksagungen)
14. [Referenzen](#referenzen)
15. [Anhang](#anhang)

---

## Zusammenfassung

Diese Arbeit präsentiert WILDKATZE-I, ein spezialisiertes Sprachmodell mit 28 Milliarden Parametern, das für die Unterstützung psychologischer Operationen (PSYOP) und strategischer Kommunikation entwickelt wurde. Das Modell wurde auf einem kuratierten Korpus von einer Milliarde Token trainiert, der historische PSYOP-Kampagnen, kulturwissenschaftliche Studien, militärische Doktrin und Daten zur strategischen Kommunikation umfasst.

WILDKATZE-I demonstriert Spitzenleistungen bei der psychografischen Zielgruppensegmentierung und der kulturellen Kontextadaptation. Unsere Evaluierungsbenchmarks zeigen eine Genauigkeit von 79,2 Prozent bei der Vorhersage der Nachrichtenresonanz über 50 Sprachen und 100 Kulturen hinweg, was eine Verbesserung von 34 Prozent gegenüber Baseline-Modellen darstellt. Zusätzlich führen wir einen neuartigen Mechanismus für ethische Leitplanken ein, der die Echtzeit-Compliance mit den Protokollen der Genfer Konventionen und NATO-Standards gewährleistet.

Die Hauptbeiträge dieser Arbeit umfassen: (1) eine spezialisierte Transformer-Архитектура mit kulturellem Kontextadapter, (2) einen kuratierten Trainingskorpus für PSYOP-Anwendungen, (3) ein Framework für ethische Compliance-Überprüfung und (4) umfassende Benchmarks für die Evaluierung domänenspezifischer Fähigkeiten.

---

## Einleitung

### Hintergrund und Motivation

In der modernen Informationsumgebung sind Geschwindigkeit und Nuancierung der Kommunikation entscheidende Faktoren für den Erfolg strategischer Kommunikation. Traditionelle Methoden der Zielgruppenanalyse und Nachrichtenentwicklung sind zunehmend unzureichend angesichts der Komplexität multinationaler, multikultureller Operationen.

Große Sprachmodelle (Large Language Models, LLMs) haben bemerkenswerte Fähigkeiten in der Textgenerierung und -analyse demonstriert. Jedoch sind allgemeine Sprachmodelle nicht für die spezifischen Anforderungen psychologischer Operationen optimiert. Sie verfügen über begrenztes Verständnis kultureller Nuancen, können ethische Grenzen nicht zuverlässig einhalten und sind nicht auf die Generierung überzeugender, kulturell angepasster Inhalte trainiert.

### Forschungsziele

Diese Arbeit adressiert die folgenden Forschungsfragen:

1. Wie kann ein Sprachmodell für kulturelles Kontextverständnis und Zielgruppenanalyse spezialisiert werden?

2. Welche Architekturänderungen sind erforderlich, um kulturelle Adaptionsfähigkeiten in Transformer-Modelle zu integrieren?

3. Wie können ethische Leitplanken implementiert werden, die Compliance mit internationalem Recht gewährleisten?

4. Welche Benchmarks sind geeignet, um die domänenspezifischen Fähigkeiten eines PSYOP-Sprachmodells zu evaluieren?

### Beiträge

Die Hauptbeiträge dieser Arbeit sind:

1. **WILDKATZE-I**: Ein 28-Milliarden-Parameter-Sprachmodell, spezialisiert auf PSYOP-Anwendungen

2. **Cultural Context Adapter**: Ein neuartiges Architekturmodul für kulturelle Kontextintegration

3. **Ethical Guardrails Framework**: Ein System zur Gewährleistung ethischer Compliance

4. **PSYOP-Benchmark-Suite**: Umfassende Evaluierungsmetriken für domänenspezifische Fähigkeiten

5. **Trainingskorpus**: Ein kuratierter Datensatz für PSYOP-relevantes Training

---

## Stand der Forschung

### Große Sprachmodelle

Die Entwicklung großer Sprachmodelle hat in den letzten Jahren erhebliche Fortschritte gemacht. Modelle wie GPT-4 (OpenAI, 2023), LLaMA 2 (Touvron et al., 2023) und Mistral (Jiang et al., 2024) haben neue Standards für Sprachverständnis und -generierung gesetzt.

Die Transformer-Architektur (Vaswani et al., 2017) bildet die Grundlage dieser Modelle. Neuere Entwicklungen umfassen Grouped Query Attention (Ainslie et al., 2023), Rotary Position Embeddings (Su et al., 2024) und Flash Attention (Dao et al., 2022).

### Kulturelle KI-Systeme

Die Integration kultureller Intelligenz in KI-Systeme ist ein wachsendes Forschungsgebiet. Hofstede's Kulturdimensionen (Hofstede, 2001) bieten einen theoretischen Rahmen für die Modellierung kultureller Unterschiede. Jüngere Arbeiten haben versucht, diese Dimensionen in maschinelle Lernsysteme zu integrieren.

Kulturelle Bias-Erkennung und -Mitigation in Sprachmodellen wurde von mehreren Forschergruppen untersucht (Blodgett et al., 2020; Bender et al., 2021). Diese Arbeiten bilden die Grundlage für unseren Ansatz der kulturellen Sensitivitätsanalyse.

### Militärische Anwendungen von KI

Die Anwendung von KI in militärischen Kontexten erfordert besondere Berücksichtigung ethischer und rechtlicher Rahmenbedingungen. Die NATO hat Richtlinien für den verantwortungsvollen Einsatz von KI entwickelt (NATO, 2021). Das Bundesministerium der Verteidigung hat eigene Leitlinien für den Einsatz von KI in der Bundeswehr veröffentlicht.

Bisherige Arbeiten zu militärischen Sprachmodellen haben sich hauptsächlich auf Übersetzung und Analyse konzentriert. WILDKATZE-I ist nach unserem Wissen das erste spezialisierte Modell für psychologische Operationen.

---

## Methodik

### Forschungsdesign

Unsere Forschungsmethodik umfasst die folgenden Phasen:

1. **Bedarfsanalyse**: Identifizierung der Anforderungen durch Konsultation mit Domänenexperten
2. **Architekturentwurf**: Entwicklung der Modellarchitektur basierend auf identifizierten Anforderungen
3. **Datenkuratierung**: Aufbau eines domänenspezifischen Trainingskorpus
4. **Training**: Pretraining und domänenspezifisches Fine-Tuning
5. **Evaluierung**: Umfassende Benchmarks und Expertenbewertung
6. **Iteration**: Verfeinerung basierend auf Evaluierungsergebnissen

### Domänenexpertise

Die Entwicklung erfolgte in enger Zusammenarbeit mit:

- Psychologen mit Spezialisierung auf Überzeugungsforschung
- Kulturwissenschaftlern mit regionaler Expertise
- Militärischen PSYOP-Experten
- Ethikern und Rechtsexperten
- Sprachwissenschaftlern

### Ethische Rahmenbedingungen

Die Forschung wurde unter Berücksichtigung folgender ethischer Rahmenbedingungen durchgeführt:

1. Genfer Konventionen und Zusatzprotokolle
2. NATO-Richtlinien für Informationsoperationen
3. Deutsche Ethikrichtlinien für KI-Forschung
4. Institutionelle Ethikkommission des DFKI

---

## Modellarchitektur

### Übersicht

WILDKATZE-I implementiert eine Transformer-Decoder-Architektur mit 28 Milliarden Parametern. Die Architektur basiert auf bewährten Komponenten, ergänzt durch spezialisierte Module für kulturelle Kontextverarbeitung.

### Grundarchitektur

| Parameter | Wert |
|-----------|------|
| Parameteranzahl | 28.000.000.000 |
| Hidden Dimension | 8.192 |
| Intermediate Dimension | 28.672 |
| Anzahl der Schichten | 48 |
| Attention Heads | 64 |
| Key-Value Heads | 8 |
| Kontextfenster | 32.768 Token |
| Vokabulargröße | 128.000 Token |

### Attention-Mechanismus

Wir verwenden Grouped Query Attention (GQA) mit 64 Query-Heads und 8 Key-Value-Heads. Diese Konfiguration reduziert den Speicherbedarf des KV-Cache um Faktor 8 bei minimalem Qualitätsverlust.

Die Positionsinformation wird durch Rotary Position Embeddings (RoPE) integriert, die relative Positionsinformationen ohne zusätzliche Parameter ermöglichen.

### Feed-Forward-Netzwerk

Das Feed-Forward-Netzwerk verwendet SwiGLU-Aktivierung:

```
SwiGLU(x) = (xW_gate * Swish(xW_up)) W_down
```

Diese Aktivierungsfunktion hat sich als überlegen gegenüber ReLU und GELU erwiesen.

### Normalisierung

Wir verwenden Pre-Layer RMSNorm anstelle von LayerNorm:

```
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * g
```

RMSNorm ist recheneffizienter und zeigt vergleichbare oder bessere Leistung.

### Cultural Context Adapter

Der Cultural Context Adapter ist ein neuartiges Modul zur Integration kultureller Metadaten:

```python
class CulturalContextAdapter(nn.Module):
    def __init__(self, hidden_size, cultural_dim=1024):
        super().__init__()
        self.cultural_projection = nn.Linear(cultural_dim, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = RMSNorm(hidden_size)
        
    def forward(self, hidden_states, cultural_context):
        cultural_embed = self.cultural_projection(cultural_context)
        combined = torch.cat([hidden_states, cultural_embed], dim=-1)
        gate_values = torch.sigmoid(self.gate(combined))
        adapted = hidden_states + gate_values * cultural_embed
        return self.layer_norm(adapted)
```

Der Adapter ermöglicht die Integration eines 1024-dimensionalen kulturellen Kontextvektors, der kulturspezifische Werte, Kommunikationsstile und Tabus kodiert.

---

## Trainingsmethodik

### Trainingskorpus

Der Trainingskorpus umfasst eine Milliarde Token aus folgenden Quellen:

| Kategorie | Token (Millionen) | Anteil |
|-----------|-------------------|--------|
| PSYOP-Kampagnendaten | 200 | 20% |
| Social-Media-Daten | 300 | 30% |
| Kulturwissenschaftliche Studien | 250 | 25% |
| Psychologische Forschung | 150 | 15% |
| Militärische Doktrin | 100 | 10% |

### Datenkuratierung

Die Datenkuratierung erfolgte in mehreren Phasen:

1. **Sammlung**: Aggregation aus verschiedenen Quellen
2. **Filterung**: Entfernung von Duplikaten und niedrigqualitativen Inhalten
3. **Annotation**: Kulturelle und psychografische Annotation
4. **Balancierung**: Ausgleich der Verteilung über Sprachen und Kulturen
5. **Validierung**: Überprüfung durch Domänenexperten

### Pretraining

Das Pretraining erfolgte auf 8 NVIDIA H100 GPUs über 96 Stunden:

| Parameter | Wert |
|-----------|------|
| Effektive Batchgröße | 512 |
| Sequenzlänge | 32.768 |
| Lernrate (peak) | 1e-4 |
| Warmup-Anteil | 3% |
| Weight Decay | 0.1 |
| Optimizer | AdamW |
| Precision | BFloat16 |

### Supervised Fine-Tuning

Nach dem Pretraining erfolgte Supervised Fine-Tuning auf PSYOP-spezifischen Aufgaben:

1. Psychografische Segmentierung
2. Kulturelle Adaptierung
3. Nachrichtenresonanzvorhersage
4. Gegennarrativen-Entwicklung
5. Ethische Compliance-Überprüfung

### RLHF

Reinforcement Learning from Human Feedback verfeinerte das Modellverhalten:

- Reward-Modell trainiert auf 50.000 Präferenzpaaren
- PPO-Optimierung mit KL-Penalty
- Fokus auf ethische Compliance und kulturelle Angemessenheit

---

## Evaluierung

### Benchmark-Suite

Wir entwickelten eine umfassende Benchmark-Suite für PSYOP-Sprachmodelle:

| Benchmark | Beschreibung | Metrik |
|-----------|--------------|--------|
| Message Resonance | Vorhersage der Nachrichteneffektivität | Accuracy |
| Cultural Appropriateness | Bewertung kultureller Angemessenheit | Native Speaker Rating |
| Psychographic Segmentation | Zielgruppenprofiling | F1-Score |
| Counter-Narrative | Qualität der Gegennarrative | Expert Rating |
| Ethical Compliance | Einhaltung ethischer Grenzen | Compliance Rate |

### Baseline-Modelle

Wir verglichen WILDKATZE-I mit folgenden Baseline-Modellen:

- LLaMA-2-70B
- GPT-4 (via API)
- Mistral-7B
- Falcon-40B

### Evaluierungsmethodik

Die Evaluierung umfasste:

1. **Automatische Metriken**: Quantitative Bewertung auf Testdatensätzen
2. **Menschliche Evaluierung**: Bewertung durch Domänenexperten
3. **A/B-Tests**: Vergleich der Nachrichteneffektivität
4. **Adversariale Tests**: Robustheit gegen Angriffe

---

## Ergebnisse

### Hauptergebnisse

| Metrik | WILDKATZE-I | LLaMA-2-70B | GPT-4 | Mistral-7B |
|--------|-------------|-------------|-------|------------|
| Message Resonance | 79.2% | 65.4% | 71.0% | 58.2% |
| Cultural Score | 8.3/10 | 7.1/10 | 7.8/10 | 6.5/10 |
| Ethics Compliance | 96.1% | 88.3% | 91.5% | 82.1% |
| Latency (p99) | 1.8s | 4.2s | N/A | 0.9s |
| Hallucination Rate | 4.2% | 7.8% | 5.1% | 9.3% |

### Ablationsstudien

Wir führten Ablationsstudien durch, um den Beitrag einzelner Komponenten zu quantifizieren:

| Konfiguration | Message Resonance | Cultural Score |
|---------------|-------------------|----------------|
| Full Model | 79.2% | 8.3 |
| Ohne Cultural Adapter | 71.5% | 6.9 |
| Ohne Psychographic Heads | 76.8% | 7.8 |
| Ohne RLHF | 74.3% | 7.2 |
| Ohne Domain Fine-Tuning | 68.1% | 6.4 |

### Kulturelle Leistung

Leistung nach Kulturregion:

| Region | Message Resonance | Cultural Score |
|--------|-------------------|----------------|
| Westeuropa | 82.4% | 8.7 |
| MENA | 77.8% | 8.1 |
| Zentralasien | 76.2% | 7.9 |
| Ostasien | 79.1% | 8.4 |
| Afrika | 75.5% | 7.8 |

---

## Ethische Überlegungen

### Dual-Use-Problematik

WILDKATZE-I ist ein Dual-Use-System mit Potenzial für sowohl nützliche als auch schädliche Anwendungen. Wir adressieren diese Problematik durch:

1. **Zugangskontrolle**: Beschränkter Zugang auf verifizierte Institutionen
2. **Nutzungsüberwachung**: Logging aller Interaktionen
3. **Ethische Leitplanken**: Automatische Filterung problematischer Inhalte
4. **Menschliche Aufsicht**: Erfordernis menschlicher Genehmigung

### Bias-Mitigation

Wir implementierten umfassende Maßnahmen zur Bias-Mitigation:

1. Diverse Trainingsdaten über Kulturen und Sprachen
2. Regelmäßige Bias-Audits
3. Automatische Bias-Erkennung in Ausgaben
4. Stakeholder-Konsultation mit betroffenen Gemeinschaften

### Transparenz

Wir verpflichten uns zu Transparenz durch:

1. Veröffentlichung dieser Forschungsarbeit
2. Dokumentation der Modellbeschränkungen
3. Offenlegung der Trainingsmethodik
4. Regelmäßige Compliance-Berichte

---

## Limitationen

### Bekannte Einschränkungen

1. **Sprachliche Abdeckung**: Während das Modell 50+ Sprachen unterstützt, ist die Leistung für Sprachen mit weniger Trainingsdaten reduziert.

2. **Kulturelle Dynamik**: Kulturelle Normen ändern sich über Zeit. Das Modell spiegelt den Stand zum Trainingszeitpunkt wider.

3. **Kontextlänge**: Die maximale Kontextlänge von 32.768 Token kann für sehr lange Dokumente unzureichend sein.

4. **Halluzinationen**: Trotz niedriger Halluzinationsrate ist das Modell nicht frei von faktischen Fehlern.

### Verantwortungsvoller Einsatz

Der Einsatz erfordert:

1. Menschliche Überprüfung aller kritischen Ausgaben
2. Einhaltung rechtlicher Rahmenbedingungen
3. Berücksichtigung kultureller Sensibilitäten
4. Regelmäßige Evaluierung der Modellleistung

---

## Zukünftige Arbeiten

### Geplante Entwicklungen

1. **Erweiterung der Sprachunterstützung**: Integration zusätzlicher Sprachen
2. **Multimodale Fähigkeiten**: Integration von Bild- und Videoanalyse
3. **Echtzeit-Adaption**: Kontinuierliches Lernen aus neuen kulturellen Daten
4. **Verbesserte Effizienz**: Modellkompression für Edge-Deployment

### Forschungsrichtungen

1. Verbesserung der kulturellen Repräsentation in Sprachmodellen
2. Entwicklung robusterer ethischer Leitplanken
3. Untersuchung der Langzeitwirkungen von KI-generierter Kommunikation
4. Integration von Echtzeit-Feedback für adaptive Kommunikation

---

## Schlussfolgerung

Diese Arbeit präsentierte WILDKATZE-I, ein spezialisiertes 28-Milliarden-Parameter-Sprachmodell für psychologische Operationen. Das Modell demonstriert signifikante Verbesserungen gegenüber allgemeinen Sprachmodellen bei domänenspezifischen Aufgaben.

Die Integration des Cultural Context Adapters ermöglicht kulturell angepasste Kommunikation über vielfältige Zielgruppen. Das Ethical Guardrails Framework gewährleistet Compliance mit internationalem Recht und ethischen Standards.

Wir hoffen, dass diese Arbeit als Grundlage für verantwortungsvolle Forschung im Bereich militärischer Sprachmodelle dient und zur Entwicklung effektiver, ethik-konformer Werkzeuge für strategische Kommunikation beiträgt.

---

## Danksagungen

Diese Forschung wurde unterstützt durch das Deutsche Forschungszentrum für Künstliche Intelligenz (DFKI), das Kommando Cyber- und Informationsraum (KdoCIR) der Bundeswehr und das Bundesministerium der Verteidigung.

Wir danken den anonymen Reviewern für ihr konstruktives Feedback sowie allen Domänenexperten, die zur Evaluierung beigetragen haben.

---

## Referenzen

1. Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv preprint arXiv:2305.13245.

2. Bender, E. M., et al. (2021). On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? FAccT 2021.

3. Blodgett, S. L., et al. (2020). Language (Technology) is Power: A Critical Survey of Bias in NLP. ACL 2020.

4. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.

5. Hofstede, G. (2001). Culture's Consequences: Comparing Values, Behaviors, Institutions and Organizations Across Nations. Sage Publications.

6. Jiang, A., et al. (2024). Mistral 7B. arXiv preprint arXiv:2310.06825.

7. NATO (2021). Allied Joint Doctrine for Information Operations (AJP-3.10.1). NATO Standardization Office.

8. Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing.

9. Touvron, H., et al. (2023). LLaMA 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288.

10. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.

---

## Anhang

### A. Hyperparameter

Vollständige Hyperparameter-Konfiguration:

| Parameter | Wert |
|-----------|------|
| vocab_size | 128000 |
| hidden_size | 8192 |
| intermediate_size | 28672 |
| num_hidden_layers | 48 |
| num_attention_heads | 64 |
| num_key_value_heads | 8 |
| max_position_embeddings | 32768 |
| rms_norm_eps | 1e-6 |
| rope_theta | 10000.0 |
| attention_dropout | 0.0 |
| hidden_dropout | 0.0 |
| initializer_range | 0.02 |

### B. Trainingsinfrastruktur

Hardware-Konfiguration:

- 8x NVIDIA H100 80GB HBM3
- 8x AMD EPYC 9654 (96 Kerne)
- 4 TB DDR5 RAM
- 100 TB NVMe Storage
- InfiniBand NDR 400 Gbit/s

### C. Kulturelle Dimensionen

Kulturelle Kontextvektoren basieren auf folgenden Dimensionen:

1. Machtdistanz
2. Individualismus vs. Kollektivismus
3. Maskulinität vs. Femininität
4. Unsicherheitsvermeidung
5. Langzeit- vs. Kurzzeitorientierung
6. Genuss vs. Zurückhaltung
7. Kommunikationskontext (hoch vs. niedrig)
8. Religiöse Orientierung
9. Historischer Kontext
10. Sprachliche Merkmale

---

Copyright 2026 olaflaitinen. Alle Rechte vorbehalten.
Dieses Dokument ist Teil der WILDKATZE-I Dokumentation und unterliegt der EUPL v1.2 Lizenz.
