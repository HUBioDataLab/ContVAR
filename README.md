# ContVAR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Graphein](https://img.shields.io/badge/Bio-Graphein-green)

**ContVAR**, tek amino asit varyantlarının (SAV) protein yapısı ve fonksiyonu üzerindeki etkisini graf tabanlı metrik öğrenmesi ile modellemek için geliştirilmiş bir projedir.

Model; aynı proteinin:

- **Anchor**: wild-type (orijinal),
- **Positive**: benign varyant,
- **Negative**: pathogenic / malign varyant

üçlülerini kullanarak gömme uzayında benign örnekleri yakına, pathogenic örnekleri uzağa itmeye çalışır.

## Mimari Özet

- **Omurga**: iki katmanlı `GATv2Conv` tabanlı `DeepProteinGAT` (`contvar/model.py`), `global_mean_pool` ile graf düzeyi gömme.
- **Kenarlar**: `ProjectConfig.edge_mode` ile `salad` (SALAD tarzı komşuluk) veya `graphein` (Graphein kenar fonksiyonları).
- **Düğüm özellikleri**: amino asit one-hot ve isteğe bağlı diğer Graphein özellikleri; ESM-2 düğüm gömmeleri (`esm_dim`, `embeddings_variable.h5` vb.).
- **Kayıp**: global triplet kaybı + mutasyon konumunda yerel kontrastif bileşen (eğitim döngüsünde birleştirilir).
- **Eğitim boru hattı** (`contvar/training.py`):
  1. **Phase 0**: GO anlamsal benzerlik ile ön eğitim (varsayılan `go_phase0_epochs` ile; DMS fazından önce çalışır).
  2. **Phase 1**: müfredat ısınması — exhaustive triplet örnekleme + standart triplet kaybı.
  3. **Phase 2**: akışkan yarı-zor negatif madencilik (`contvar/mining.py`) + yarı-zor triplet kaybı.
- **İzleme**: Weights & Biases (`wandb`), isteğe bağlı t-SNE (`--visualize`).

## Depo Yapısı

```text
ContVAR/
├── train.py                 # Yerel CLI giriş noktası
├── run.ipynb                # Colab odaklı not defteri
├── setup.py                 # Paket kurulumu (bağımlılık listesi yok; aşağıdaki pip satırına bakın)
├── contvar/
│   ├── config.py            # ProjectConfig, setup_environment, Colab zip yolu
│   ├── model.py             # DeepProteinGAT
│   ├── training.py          # train_pipeline: Phase 0 → DMS müfredat eğitimi
│   ├── go_pretraining.py    # Phase-0 GO ön eğitimi
│   ├── go_identity_split.py # protein_to_split JSON yükleme ve triplet filtreleme
│   ├── losses.py          # Standard / semi-hard triplet
│   ├── mining.py            # streaming semi-hard madencilik
│   ├── metrics.py           # AUC, mesafe metrikleri
│   ├── edges.py             # SALAD tarzı kenar üretimi
│   ├── utils.py             # ESM h5 yükleme
│   ├── viz_tsne.py          # t-SNE görselleştirme
│   ├── viz_graph.py         # örnek graf görselleştirme / wandb
│   └── data/
│       ├── mapper.py        # triplet yolları, train/val split JSON
│       ├── dataset.py       # CIF → PyG (DMS), ExhaustiveTripletDataset
│       ├── collate.py       # triplet batch birleştirme
│       └── go_dataset.py    # Phase-0: yalnızca önceden üretilmiş .pt grafikleri
├── graph_prebuilder/        # GO için toplu .pt graf üretimi (build_all_graphs.py, README)
├── local_splits/            # Phase-0 protein bazlı train/val/test (ör. graphless çıkarılmış birleşik JSON)
├── semantic_similarity/     # GO TSV’leri (MF/BP/CC filtreli benzerlik tabloları)
├── protein_triplets_data/   # originals / positives / negatives CIF hiyerarşisi (DMS)
└── *.md                     # PHASE0_GO_PRETRAINING.md, PHASE0_TASK_BRIEF.md, toplantı notları
```

## Kurulum

`setup.py` içinde `install_requires` tanımlı değildir; aşağıdaki paketleri manuel kurun:

```bash
pip install -e .
pip install torch torch-geometric graphein wandb biopython h5py scikit-learn matplotlib tqdm numpy networkx
```

Graphein genelde `pandas` gibi dolaylı bağımlılıkları çeker; eksik modül hatası alırsanız pip çıktısına göre tamamlayın.

## Veri Gereksinimleri

### DMS triplet eğitimi

- **`protein_triplets_data/`**: `originals/` altında doğrudan `.cif`; `positives/<protein_id>/` ve `negatives/<protein_id>/` alt klasörlerinde varyant yapıları.
- **`embeddings_variable.h5`** (veya eşdeğer): ESM-2 düğüm gömmeleri; `train.py` ile `--embeddings` veya `setup_environment` varsayılanlarıyla verilir.

### Phase-0 GO ön eğitimi

Tam eğitim için Phase-0 verileri ve önceden üretilmiş graf dizini gereklidir (`go_prebuilt_graph_root`).

- **`semantic_similarity/`** altında beklenen dosya adları (kodda sabit):

  - `semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv`
  - `semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv`
  - `semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv`

- **Önceden üretilmiş PyG graf dizini**: `GOSemanticTripletDataset` yalnızca `.pt` dosyalarından okur; CIF’ten anlık kurulum yoktur. Graf üretimi için `graph_prebuilder/build_all_graphs.py` ve klasördeki `README.md` kullanılır.
- **Protein split**: `local_splits/phase0_protein_split_removed_graphless.json` — üst düzey `protein_to_split` anahtarı ile protein → `train` | `val` | `test` eşlemesi. Varsayılan yol `ProjectConfig.go_protein_split_json_path` içindedir.

## Hızlı Başlangıç

1. **Phase-0 grafları**: `graph_prebuilder/build_all_graphs.py` (veya aynı formatta `.pt` üreten süreç) ile ilgili proteinler için PyG graf dosyalarını hazırlayın.
2. **Yapılandırma**: `contvar/config.py` içinde `go_prebuilt_graph_root` değerini bu graf dizinine ayarlayın (mutlak yol önerilir). `semantic_similarity` TSV’leri ve `local_splits` altındaki protein split JSON’u (`go_protein_split_json_path`) yerinde olmalıdır.
3. **Çalıştırma**: DMS verisi ve ESM gömmeleriyle birlikte eğitim:

```bash
python train.py --data-root protein_triplets_data --embeddings embeddings_variable.h5
```

Özet akış: önce Phase-0 (GO), ardından DMS triplet müfredatı (Phase 1 ve 2). `go_phase0_epochs` ve diğer Phase-0 hiperparametreleri `ProjectConfig` üzerinden ayarlanır.

## CLI (`train.py`)

| Argüman | Açıklama |
|--------|----------|
| `--data-root` | `protein_triplets_data` kökü |
| `--embeddings` | ESM-2 `h5` yolu |
| `--force` | Graf önbelleğini / işlenmiş veriyi sıfırdan üret |
| `--split-path` | Önceden kaydedilmiş split JSON (yeniden üretilebilirlik) |
| `--wandb-key` | WandB API anahtarı (veya `WANDB_API_KEY` ortam değişkeni) |
| `--visualize` | Eğitim sonunda validation gömmeleri için t-SNE |

Checkpoint’ler çalışma dizinine yazılır:

- `model_best_loss.pt`
- `model_last.pt`

Triplet split dosyası `mapper` tarafından `data_root` altında saklanır (mantık: `contvar/data/mapper.py`).

## Phase-0: Ontoloji Örnekleme (GOAL2)

Phase-0’da MF/BP/CC dengesi, kayıp ağırlığı yerine **örnekleme oranı** ile kontrol edilir. Varsayılanlar:

```python
"go_sampling_enabled": True,
"go_sampling_ratio": {"mf": 0.6, "bp": 0.2, "cc": 0.2},
"go_log_sampling_stats": True,
```

Tamamı `ProjectConfig` üzerinden değiştirilebilir.

## Konfigürasyon

Tüm temel hiperparametreler ve yollar `contvar/config.py` içindeki **`ProjectConfig`** sınıfında toplanır: kenar modu, epoch, margin, batch boyutları, müfredat (warmup epoch sayısı), Phase-1 erken durdurma penceresi, GO TSV dizini, `go_max_triplets_per_ontology`, `go_prebuilt_graph_root`, vb.

`train_pipeline(..., config={...})` ile sözlük override (ör. WandB sweep) desteklenir; yerel `train.py` şu an bu sözlüğü CLI’dan geçirmez — deney parametreleri için `run.ipynb`, WandB veya `config.py` düzenlemesi kullanılır.

## Ek Belgeler

- **`PHASE0_GO_PRETRAINING.md`**: Phase-0 davranışı ve parametreler (bazı bölümler tarihsel olabilir; graf yükleme tarafında güncel kaynak kod `contvar/data/go_dataset.py` ve `go_pretraining.py`dır).
- **`PHASE0_TASK_BRIEF.md`**, **`GO_DMS_TOPLANTI_NOTU_*.md`**: görev ve toplantı notları.
- **`graph_prebuilder/README.md`**: toplu `.pt` üretimi ve Colab not defteri.

## Notlar

- Gizli anahtarları repoya yazmayın; `WANDB_API_KEY` ortam değişkenini kullanın.
- Colab için varsayılan Drive yolları `setup_environment` ve `run.ipynb` içinde tanımlıdır; yerelde `train.py` daha doğrudan bir başlangıçtır.
