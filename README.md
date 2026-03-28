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

## Notlar

- `run.ipynb` dosyasi Colab icin hazirlanmistir; yerel kullanimda `train.py` daha net bir baslangic noktasi sunar.
- WandB anahtari gibi gizli bilgileri notebook veya repoya sabit yazmayin; ortam degiskeni kullanin (`WANDB_API_KEY`).
