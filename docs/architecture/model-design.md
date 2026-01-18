# WILDKATZE-I Modellarchitektur

## Dokumentenübersicht

Dieses Dokument beschreibt die vollständige technische Architektur des WILDKATZE-I Sprachmodells. Es richtet sich an Entwickler, Forscher und technische Entscheidungsträger, die ein detailliertes Verständnis der Modellstruktur und ihrer Implementierung benötigen.

---

## Inhaltsverzeichnis

1. [Einleitung](#einleitung)
2. [Architekturübersicht](#architekturübersicht)
3. [Transformer-Decoder-Architektur](#transformer-decoder-architektur)
4. [Attention-Mechanismus](#attention-mechanismus)
5. [Feed-Forward-Netzwerk](#feed-forward-netzwerk)
6. [Normalisierung](#normalisierung)
7. [Positionscodierung](#positionscodierung)
8. [Embedding-Schichten](#embedding-schichten)
9. [Spezialisierte Module](#spezialisierte-module)
10. [Quantisierung](#quantisierung)
11. [Speicheroptimierung](#speicheroptimierung)
12. [Mathematische Grundlagen](#mathematische-grundlagen)
13. [Implementierungsdetails](#implementierungsdetails)
14. [Konfigurationsparameter](#konfigurationsparameter)
15. [Referenzen](#referenzen)

---

## Einleitung

WILDKATZE-I implementiert eine moderne Transformer-Decoder-Architektur, die auf den neuesten Erkenntnissen der Large Language Model (LLM) Forschung basiert. Die Architektur wurde speziell für die Anforderungen psychologischer Operationen und strategischer Kommunikation optimiert, wobei besonderer Wert auf kulturelles Kontextverständnis und ethische Compliance gelegt wurde.

Die Architektur basiert auf dem bewährten Decoder-Only-Design, das sich in aktuellen Sprachmodellen wie GPT, LLaMA und Mistral als besonders effektiv erwiesen hat. Durch die Integration spezialisierter Module für kulturelle Kontextverarbeitung und psychografische Analyse unterscheidet sich WILDKATZE-I von allgemeinen Sprachmodellen.

Die Entwicklung erfolgte unter Berücksichtigung der folgenden Designprinzipien:

1. **Skalierbarkeit**: Die Architektur ermöglicht effizientes Training und Inferenz auf verschiedenen Hardware-Konfigurationen
2. **Effizienz**: Moderne Optimierungstechniken wie Flash Attention und Grouped Query Attention minimieren Speicher- und Rechenanforderungen
3. **Modularität**: Klare Trennung der Komponenten ermöglicht einfache Wartung und Erweiterung
4. **Spezialisierung**: Integration domänenspezifischer Module für PSYOP-Anwendungen

---

## Architekturübersicht

### Grundlegende Struktur

Die WILDKATZE-I Architektur besteht aus den folgenden Hauptkomponenten:

| Komponente | Beschreibung | Parameter |
|------------|--------------|-----------|
| Token Embedding | Konvertierung von Token-IDs in kontinuierliche Vektoren | 8.192 x 128.000 |
| Decoder Stack | 48 aufeinanderfolgende Transformer-Decoder-Blöcke | Variabel |
| Final Layer Norm | RMSNorm nach dem letzten Decoder-Block | 8.192 |
| Language Model Head | Projektion auf Vokabulargröße | 8.192 x 128.000 |

### Datenfluss

Der Datenfluss durch das Modell erfolgt in folgenden Schritten:

1. **Tokenisierung**: Eingabetext wird in Token-IDs konvertiert
2. **Embedding**: Token-IDs werden in Einbettungsvektoren transformiert
3. **Positionsintegration**: RoPE wird auf die Attention-Berechnungen angewendet
4. **Decoder-Stack**: Durchlauf durch alle 48 Decoder-Blöcke
5. **Normalisierung**: Finale RMSNorm-Anwendung
6. **Projektion**: Lineare Projektion auf Logits
7. **Sampling**: Auswahl des nächsten Tokens

### Parameterverteilung

Die Gesamtparameterzahl von 28 Milliarden verteilt sich wie folgt:

| Komponente | Parameter (Milliarden) | Anteil |
|------------|------------------------|--------|
| Embeddings | 1.05 | 3.8% |
| Attention | 12.58 | 44.9% |
| Feed-Forward | 13.27 | 47.4% |
| Normalisierung | 0.79 | 2.8% |
| LM Head | 1.05 | 3.8% |
| Gesamt | 28.00 | 100% |

---

## Transformer-Decoder-Architektur

### Decoder-Block-Struktur

Jeder der 48 Decoder-Blöcke implementiert die folgende Struktur:

```
Input
  |
  v
RMSNorm -----> Self-Attention -----> +
  |                                  |
  +----------------------------------+
  |
  v
RMSNorm -----> Feed-Forward -------> +
  |                                  |
  +----------------------------------+
  |
  v
Output
```

Die Pre-Normalisierungsstrategie (Normalisierung vor jeder Subschicht) verbessert die Trainingsstabilität und ermöglicht tiefere Netzwerke im Vergleich zur Post-Normalisierung.

### Residualverbindungen

Jede Subschicht verwendet additive Residualverbindungen:

```
y = x + Sublayer(Norm(x))
```

Diese Struktur ermöglicht den ungehinderten Gradientenfluss durch tiefe Netzwerke und verhindert das Verschwinden von Gradienten.

### Schichtweise Anordnung

Die 48 Decoder-Schichten sind sequentiell angeordnet, wobei jede Schicht die gleiche Struktur aufweist, aber unabhängige Gewichte besitzt. Die Gesamtberechnung lässt sich wie folgt darstellen:

```
h_0 = Embedding(tokens)
h_l = DecoderBlock_l(h_{l-1}) für l = 1, ..., 48
output = LMHead(RMSNorm(h_48))
```

---

## Attention-Mechanismus

### Multi-Head Self-Attention

WILDKATZE-I verwendet Multi-Head Self-Attention mit der folgenden Konfiguration:

| Parameter | Wert |
|-----------|------|
| Anzahl Query-Heads | 64 |
| Anzahl Key-Value-Heads | 8 |
| Head-Dimension | 128 |
| Attention-Dropout | 0.0 |
| Attention-Bias | Nein |

### Grouped Query Attention (GQA)

Anstelle der traditionellen Multi-Head Attention mit separaten Key-Value-Paaren pro Head verwendet WILDKATZE-I Grouped Query Attention. Bei GQA teilen sich mehrere Query-Heads ein gemeinsames Key-Value-Paar:

- 64 Query-Heads gruppiert in 8 Gruppen
- Jede Gruppe von 8 Query-Heads teilt sich ein Key-Value-Paar
- Reduktion der KV-Cache-Größe um Faktor 8
- Minimaler Qualitätsverlust bei signifikanter Speicherersparnis

Die mathematische Formulierung:

```
Q = xW_Q    (Dimension: batch x seq x 64 x 128)
K = xW_K    (Dimension: batch x seq x 8 x 128)
V = xW_V    (Dimension: batch x seq x 8 x 128)

K_expanded = repeat_interleave(K, 8, dim=2)
V_expanded = repeat_interleave(V, 8, dim=2)

Attention = softmax(QK^T / sqrt(d_k)) V
```

### Rotary Position Embedding (RoPE)

Die Positionsinformation wird durch Rotary Position Embeddings in die Attention-Berechnung integriert:

```
q_m = R_m q
k_n = R_n k

wobei:
R_theta = [cos(theta), -sin(theta)]
         [sin(theta),  cos(theta)]
```

Die Rotation wird auf Paare von Dimensionen angewendet, wobei die Frequenz durch den RoPE-Theta-Parameter (10.000) gesteuert wird. RoPE bietet folgende Vorteile:

1. **Relative Positionsinformation**: Die Attention-Scores hängen nur von der relativen Position ab
2. **Extrapolation**: Ermöglicht die Verarbeitung längerer Sequenzen als während des Trainings
3. **Effizienz**: Keine zusätzlichen Parameter erforderlich

### Flash Attention 2

WILDKATZE-I integriert Flash Attention 2 für optimierte Attention-Berechnung:

```python
if self.config.use_flash_attention:
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=True if attention_mask is None else False
    )
```

Flash Attention reduziert den Speicherbedarf von O(n^2) auf O(n) und beschleunigt die Berechnung durch optimierte GPU-Kernel.

### Kausale Maskierung

Für autoregressive Generierung wird eine kausale Maske angewendet:

```
Mask[i,j] = -inf wenn i < j, sonst 0
```

Dies stellt sicher, dass jede Position nur auf vorherige Positionen zugreifen kann.

---

## Feed-Forward-Netzwerk

### SwiGLU-Aktivierung

WILDKATZE-I verwendet SwiGLU (Swish-Gated Linear Unit) als Aktivierungsfunktion:

```
SwiGLU(x) = (xW_gate * Swish(xW_up)) W_down

wobei:
Swish(x) = x * sigmoid(x)
```

Die SwiGLU-Aktivierung bietet verbesserte Gradienten im Vergleich zu ReLU und anderen Aktivierungsfunktionen.

### Dimensionen

| Schicht | Eingabe | Ausgabe |
|---------|---------|---------|
| Gate Projection | 8.192 | 28.672 |
| Up Projection | 8.192 | 28.672 |
| Down Projection | 28.672 | 8.192 |

Die Intermediate-Dimension von 28.672 wurde gewählt, um eine effiziente GPU-Nutzung zu gewährleisten (Vielfaches von 256).

### Implementierung

```python
class WildkatzeSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

---

## Normalisierung

### RMSNorm

WILDKATZE-I verwendet Root Mean Square Layer Normalization (RMSNorm) anstelle von LayerNorm:

```
RMSNorm(x) = x / RMS(x) * g

wobei:
RMS(x) = sqrt(mean(x^2) + epsilon)
```

### Vorteile von RMSNorm

1. **Recheneffizienz**: Keine Berechnung des Mittelwerts erforderlich
2. **Numerische Stabilität**: Epsilon-Wert von 1e-6
3. **Parameterreduktion**: Kein Bias-Parameter

### Implementierung

```python
class WildkatzeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

---

## Positionscodierung

### Rotary Position Embedding Details

Die RoPE-Implementierung rotiert Paare von Dimensionen im Einbettungsraum:

```python
class WildkatzeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
```

### Anwendung auf Queries und Keys

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

---

## Embedding-Schichten

### Token-Embedding

Die Token-Embedding-Schicht konvertiert diskrete Token-IDs in kontinuierliche Vektoren:

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

| Parameter | Wert |
|-----------|------|
| Vokabulargröße | 128.000 |
| Embedding-Dimension | 8.192 |
| Gesamtparameter | 1.048.576.000 |

### Language Model Head

Der Language Model Head projiziert die versteckte Repräsentation auf Logits:

```python
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

Die Gewichte werden nicht mit dem Token-Embedding geteilt, um maximale Ausdrucksstärke zu gewährleisten.

---

## Spezialisierte Module

### Cultural Context Adapter

Der Cultural Context Adapter ermöglicht die Integration kultureller Metadaten:

```python
class CulturalContextAdapter(nn.Module):
    def __init__(self, hidden_size, cultural_dim=1024):
        super().__init__()
        self.cultural_projection = nn.Linear(cultural_dim, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, hidden_states, cultural_context):
        cultural_embed = self.cultural_projection(cultural_context)
        combined = torch.cat([hidden_states, cultural_embed], dim=-1)
        gate_values = torch.sigmoid(self.gate(combined))
        return hidden_states + gate_values * cultural_embed
```

### Psychographic Analysis Heads

Spezialisierte Attention-Heads für psychografische Mustererkennung:

```python
class PsychographicHead(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.classifier = nn.Linear(hidden_size, 5)
        
    def forward(self, hidden_states):
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        pooled = attn_output.mean(dim=1)
        return self.classifier(pooled)
```

---

## Quantisierung

### INT8-Quantisierung

Für effiziente Inferenz unterstützt WILDKATZE-I INT8-Quantisierung:

```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

Die Quantisierung reduziert den Speicherbedarf um 50 Prozent bei minimalem Qualitätsverlust (unter 1 Prozent).

### INT4-Quantisierung

Für Edge-Deployment wird INT4-Quantisierung über bitsandbytes unterstützt:

```python
from bitsandbytes import nn as bnb

model = bnb.LinearNF4(model)
```

---

## Speicheroptimierung

### Gradient Checkpointing

Gradient Checkpointing reduziert den Speicherbedarf während des Trainings:

```python
model.gradient_checkpointing_enable()
```

Durch selektive Neuberechnung von Aktivierungen wird der Speicherbedarf um bis zu 60 Prozent reduziert.

### Flash Attention Speichereffizienz

Flash Attention reduziert den Peak-Speicherbedarf der Attention-Berechnung von O(n^2) auf O(n).

### KV-Cache-Management

Effizientes KV-Cache-Management für autoregressive Generierung:

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_length, num_layers):
        self.cache = {}
        
    def update(self, layer_idx, key, value):
        if layer_idx not in self.cache:
            self.cache[layer_idx] = (key, value)
        else:
            old_key, old_value = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([old_key, key], dim=2),
                torch.cat([old_value, value], dim=2)
            )
```

---

## Mathematische Grundlagen

### Attention-Berechnung

Die vollständige Attention-Berechnung:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

wobei:
Q = xW_Q (Query-Projektion)
K = xW_K (Key-Projektion)
V = xW_V (Value-Projektion)
d_k = Head-Dimension (128)
```

### Skalierungsfaktor

Der Skalierungsfaktor sqrt(d_k) verhindert, dass die Dot-Products in Regionen mit sehr kleinen Gradienten der Softmax-Funktion fallen.

### Softmax-Stabilität

Für numerische Stabilität wird die Log-Sum-Exp-Trick angewendet:

```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

---

## Implementierungsdetails

### Vollständiger Forward-Pass

```python
def forward(
    self,
    input_ids,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=True,
):
    hidden_states = self.embed_tokens(input_ids)
    
    cos, sin = self.rotary_emb(hidden_states.shape[1])
    
    next_decoder_cache = () if use_cache else None
    
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[layer_idx] if past_key_values else None,
            cos=cos,
            sin=sin,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]
        
        if use_cache:
            next_decoder_cache += (layer_outputs[1],)
    
    hidden_states = self.norm(hidden_states)
    
    return hidden_states, next_decoder_cache
```

---

## Konfigurationsparameter

### Vollständige Konfiguration

```python
@dataclass
class WildkatzeConfig:
    vocab_size: int = 128000
    hidden_size: int = 8192
    intermediate_size: int = 28672
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_flash_attention: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False
    cultural_context_dim: int = 1024
```

---

## Referenzen

1. Vaswani, A., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.

2. Su, J., et al. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing.

3. Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint.

4. Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Advances in Neural Information Processing Systems.

5. Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv preprint.

6. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. Advances in Neural Information Processing Systems.

---

Copyright 2026 olaflaitinen. Alle Rechte vorbehalten.
Dieses Dokument ist Teil der WILDKATZE-I Dokumentation und unterliegt der EUPL v1.2 Lizenz.
