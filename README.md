# ContVAR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Graphein](https://img.shields.io/badge/Bio-Graphein-green)

**ContVAR**, tek amino asit varyantlarinin (SAV) protein yapisi ve fonksiyonu uzerindeki etkisini graph tabanli metric learning ile modellemek icin gelistirilmis bir projedir.

Model; ayni proteinin:
- **Anchor**: Wild-type (orijinal),
- **Positive**: benign varyant,
- **Negative**: pathogenic/malignant varyant

uclulerini kullanarak embedding uzayinda benign ornekleri yakina, pathogenic ornekleri uzaga itmeye calisir.

## Proje Yapisi

Ana dosyalar:
- `train.py`: egitim giris noktasi (CLI), opsiyonel t-SNE cizimi.
- `run.ipynb`: Colab odakli calisma not defteri.
- `contvar/config.py`: tum temel ayarlar (`ProjectConfig`) ve ortam kurulumu.
- `contvar/model.py`: `DeepProteinGAT` modeli.
- `contvar/training.py`: egitim dongusu, degerlendirme, checkpoint kaydi.
- `contvar/go_pretraining.py`: GO semantic similarity ile Phase-0 pretraining.

## Veri Seti Yapisi

Kodun bekledigi varsayilan yapi:

```text
ContVAR/
├── protein_triplets_data/
│   ├── originals/
│   │   ├── 1abc.cif
│   │   └── 2xyz.cif
│   ├── positives/
│   │   ├── 1abc/
│   │   │   ├── 1abc_var1.cif
│   │   │   └── 1abc_var2.cif
│   │   └── 2xyz/
│   └── negatives/
│       ├── 1abc/
│       │   ├── 1abc_bad1.cif
│       │   └── 1abc_bad2.cif
│       └── 2xyz/
└── embeddings_variable.h5
```

Notlar:
- `originals` altinda dosyalar dogrudan bulunur.
- `positives` ve `negatives` altinda her protein ID icin ayri klasor bulunur.
- Mevcut kod akisi `.cif` dosyalari ile calisacak sekilde kurgulanmistir.

## Kurulum

`setup.py` icinde bagimliliklar sabitlenmedigi icin paketleri manuel kurmaniz gerekir.

Ornek kurulum:

```bash
pip install -e .
pip install torch torch-geometric graphein wandb biopython h5py scikit-learn matplotlib pandas networkx tqdm
```

## Egitim

Yerel calistirma:

```bash
python train.py --data-root protein_triplets_data --embeddings embeddings_variable.h5
```

Sik kullanilan argumanlar:
- `--force`: graph cache/split yeniden olusturma.
- `--split-path <path>`: ozel split dosyasi.
- `--wandb-key <key>`: WANDB API key.
- `--visualize`: egitim sonunda validation embeddingleri icin t-SNE olusturur.

Checkpoint dosyalari varsayilan olarak calisma dizinine yazilir:
- `model_best_loss.pt`
- `model_last.pt`

## Model ve Egitim Akisi

Ozet:
- Omurga: GATv2 tabanli `DeepProteinGAT`.
- Pooling: `global_mean_pool`.
- Kayip: triplet + yerel contrastive bilesenler (konfige gore).
- Mining: semi-hard negative mining.
- Curriculum: egitim fazlara bolunebilir.
- Opsiyonel: GO semantic similarity tabanli Phase-0 pretraining.

Bu nedenle proje yalnizca "tek bir TripletMarginLoss adimi"ndan ibaret degildir; egitim boru hatti daha kapsamli bir sekilde tasarlanmistir.

## Konfigurasyon

Temel ayarlar `contvar/config.py` icindeki `ProjectConfig` ile yonetilir. Ornek alanlar:
- graph edge modu (`salad` / `graphein`),
- epoch, margin, batch size,
- ESM embedding kullanimi (`esm_dim`),
- GO Phase-0 dosya ve epoch ayarlari.

## Phase-0 GO: kimlik bilincine sahip train/val/test

GO TSV’lerindeki `sim_mf` / `sim_bp` / `sim_cc` **semantic** (anlamsal) benzerliktir; sekans kimligi (> %50) icin TSV’deki skorlar kullanilmaz. Train/val/test ayrimi **grup bazinda** yapilir (ornegin UniProt protein ID → UniRef50 cluster mapping’i ile): ayni gruptaki proteinler ayni bolmede kalir; `80 / 10 / 10` orani **grup sayisina** gore uygulanir.

- Harici mapping: `protein_id` ve `group_id` (veya `uniref`, `cluster_id`) kolonlu tab-ayirali dosya; `go_cluster_map_path`.
- Paylasilabilir split ciktisi: `go_split_json_path` (icinde `group_id -> train|val|test`). Diger ekip ayni JSON + ayni mapping ile es split’i alir. Bu dosyalari repoya commit etmeyin; Drive/zip/e-posta ile paylasin (`.gitignore` ornekleri: `local_splits/`, `*_phase0_go_split.json`).
- Offline uretim: `python scripts/build_phase0_go_split.py --cluster-map ... --mf-tsv ... --bp-tsv ... --cc-tsv ... --output phase0_go_split.json` (Torch gerektirmez; `contvar.go_identity_split` kullanir).

### Phase-0 split JSON’u tek seferde uretmek (onerilen)

Proje kokunden (internet gerekir; buyuk TSV’lerde UniProt sorgulari uzun surebilir):

```bash
python scripts/generate_phase0_split_bundle.py
```

Bu komut sirasiyla:

1. GO TSV’lerde gecen tum UniProt ID’leri icin UniProt REST API ile `UniRef50` cluster ID’lerini indirir ve `local_splits/protein_uniref50.tsv` yazar.
2. `local_splits/phase0_go_split.json` dosyasini (grup -> train/val/test, 80/10/10) uretir.

Zaten elinizde `protein_id` / `group_id` TSV’si varsa sadece JSON uretin:

```bash
python scripts/generate_phase0_split_bundle.py --skip-fetch --cluster-map /path/to/mapping.tsv --output-json /path/to/phase0_go_split.json
```

Sadece mapping indirmek icin:

```bash
python scripts/fetch_uniref50_cluster_map.py --output local_splits/protein_uniref50.tsv --mf-tsv semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_mf.tsv --bp-tsv semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_bp.tsv --cc-tsv semantic_similarity/semantic_similarity_swissprot_filtered_low0.2_high0.8_cc.tsv
```

Egitimde ornek:

```python
"go_split_mode": "identity_grouped",
"go_cluster_map_path": r"C:\path\to\ContVAR\local_splits\protein_uniref50.tsv",
"go_split_json_path": r"C:\path\to\ContVAR\local_splits\phase0_go_split.json",
```

Config anahtarlari: `go_split_mode` (`none` | `identity_grouped`), `go_split_seed`, `go_train_ratio`, `go_val_ratio`, `go_test_ratio`, `go_cluster_map_path`, `go_split_json_path`, `go_save_split_json_path`.

## Phase-0 GO: ontology sampling orani (GOAL2)

Phase-0'da MF/BP/CC dengesi loss katsayisi ile degil, sampling ile kontrol edilir. Varsayilan oran `mf=0.6`, `bp=0.2`, `cc=0.2` olarak gelir; ancak tamamen config'den degistirilebilir:

```python
"go_sampling_enabled": True,
"go_sampling_ratio": {"mf": 0.6, "bp": 0.2, "cc": 0.2},
"go_log_sampling_stats": True,
```

Boylece "hangi ontolojiden ne oranda sample aliyoruz?" sorusunun cevabi sabit kod yerine dogrudan konfigrasyonda tutulur; deneyden deneye kolayca override edilebilir.

## Notlar

- `run.ipynb` dosyasi Colab icin hazirlanmistir; yerel kullanimda `train.py` daha net bir baslangic noktasi sunar.
- WandB anahtari gibi gizli bilgileri notebook veya repoya sabit yazmayin; ortam degiskeni kullanin (`WANDB_API_KEY`).
