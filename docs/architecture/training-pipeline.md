# Trainingspipeline

## Überblick

Die WILDKATZE-I Trainingspipeline ist für verteiltes Training auf GPU-Clustern optimiert.

## Phasen

### 1. Pretraining

- **Daten**: 1 Milliarde Token aus PSYOP, kulturellen Studien, militärischer Doktrin
- **Hardware**: 8x H100 GPUs (minimum)
- **Precision**: BFloat16
- **Optimizer**: AdamW (beta1=0.9, beta2=0.95)

### 2. Supervised Fine-Tuning (SFT)

- Feinabstimmung auf PSYOP-spezifische Aufgaben
- LoRA für effizientes Training

### 3. RLHF

- Reinforcement Learning from Human Feedback
- Optimierung auf ethische Compliance

## Konfiguration

Siehe `configs/training/` für vollständige Konfigurationsdateien.
