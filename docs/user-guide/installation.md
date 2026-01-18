# Installation

## Systemanforderungen

### Hardware
- CPU: 16+ Kerne empfohlen
- RAM: 64 GB+ für Inferenz
- GPU: NVIDIA A100/H100 (Training), RTX 4090+ (Inferenz)
- Speicher: 100 GB+

### Software
- Python 3.10+
- CUDA 12.1+
- PyTorch 2.1+

## Installationsanleitung

### Schnellinstallation

```bash
git clone https://github.com/olaflaitinen/wildkatze-i.git
cd wildkatze-i
pip install -e .
```

### Vollständige Installation (Entwicklung)

```bash
git clone https://github.com/olaflaitinen/wildkatze-i.git
cd wildkatze-i
./scripts/setup/setup_environment.sh
source .venv/bin/activate
```

### Docker-Installation

```bash
docker pull olaflaitinen/wildkatze-api:latest
docker run -p 8080:8080 olaflaitinen/wildkatze-api:latest
```

## Verifizierung

```bash
python -c "from wildkatze import WildkatzeConfig; print(WildkatzeConfig())"
```
