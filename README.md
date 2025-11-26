# ContVAR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Graphein](https://img.shields.io/badge/Bio-Graphein-green)

**ContVAR**, proteinlerin tek amino asit varyantlarının (SAVs) yapısal ve fonksiyonel etkilerini analiz etmek için geliştirilmiş yapay zeka tabanlı bir projedir.

Proje, proteinlerin 3 boyutlu PDB yapılarını graflara dönüştürür ve **Metric Learning (Triplet Loss)** yaklaşımını kullanarak; hastalığa neden olan (malignant) varyantları, zararsız (benign) varyantlardan uzamsal olarak ayrıştırmayı hedefler.

## 🧬 Proje Amacı

Protein dizilimindeki tek bir harf değişikliği (mutasyon), proteinin yapısını bozabilir veya etkisiz kalabilir. ContVAR, bu değişimleri sadece dizi (sequence) üzerinde değil, **3 boyutlu uzaydaki komşuluk ilişkilerini** de gözeterek analiz eder.

Model şu mantıkla eğitilir:
1.  **Anchor (Çapa):** Proteinin orijinal hali (Wild-Type).
2.  **Positive (Pozitif):** Aynı proteinin zararsız (Benign) varyantı.
3.  **Negative (Negatif):** Aynı proteinin hastalık yapan (Malignant/Pathogenic) varyantı.

Amaç, embedding uzayında "Benign" varyantı orijinal proteine yakın tutarken, "Malignant" varyantı onlardan uzaklaştırmaktır.

## 📂 Veri Seti Yapısı (Directory Structure)

Projenin çalışabilmesi için veri setinin aşağıdaki hiyerarşide olması gerekmektedir. Kod, `protein_triplets_data` klasörünü baz alır.

```text
ContVAR/
│
├── protein_triplets_data/
│   ├── originals/               # Wild-Type (Referans) Proteinler
│   │   ├── 1abc.pdb             # Örn: Orijinal protein yapısı
│   │   └── 2xyz.pdb
│   │
│   ├── positives/               # Benign (Zararsız) Varyantlar
│   │   ├── 1abc/                # DİKKAT: Klasör adı original ID ile aynı olmalı
│   │   │   ├── 1abc_var1.pdb
│   │   │   └── 1abc_var2.pdb
│   │   └── 2xyz/
│   │       └── ...
│   │
│   └── negatives/               # Malignant (Hastalık Yapan) Varyantlar
│       ├── 1abc/                # DİKKAT: Klasör adı original ID ile aynı olmalı
│       │   ├── 1abc_bad1.pdb
│       │   └── 1abc_bad2.pdb
│       └── 2xyz/
│           └── ...
│
├── graphein.ipynb (veya .py)
└── README.md
```

* **originals:** Sadece `.pdb` dosyalarını içerir.
* **positives & negatives:** İçlerinde her protein ID'si için ayrı bir **klasör** bulundurur. Varyant `.pdb` dosyaları bu alt klasörlerde yer alır.

## ⚙️ Teknik Detaylar ve Graphein Konfigürasyonu

Bu projede biyolojik yapıları grafa dönüştürmek için **Graphein** kütüphanesi kullanılmıştır. Modelin proteinleri nasıl "gördüğü" aşağıdaki parametrelerle belirlenmiştir:

### 1. Graf Oluşturma (Graph Construction)
* **Düğüm Özellikleri (Node Features):** `amino_acid_one_hot` kullanılmıştır. Her düğüm (amino asit), 20 boyutlu bir vektörle temsil edilir. Bu sayede model, mutasyonun türünü (örneğin Alanin -> Triptofan değişimini) net bir şekilde ayırt edebilir.
* **Kenar Oluşturma (Edge Construction):**
    * `add_peptide_bonds`: Protein omurgasını (backbone) korumak için ardışık amino asitler bağlanır.
    * `add_k_nn_edges (k=10)`: Proteinin 3 boyutlu katlanmasını modele öğretmek için kullanılır. Uzayda birbirine en yakın 10 amino asit, dizide birbirlerinden uzak olsalar bile bağlanır. Bu, mutasyonun çevresindeki **mikro-çevreyi** analiz etmek için kritiktir.

### 2. Model Mimarisi: DeepProteinGAT
Model, **GATv2 (Graph Attention Network v2)** mimarisi üzerine kurulmuştur:
* **Attention:** Mutasyonun komşu amino asitlerle etkileşim ağırlıklarını öğrenir.
* **Pooling:** `global_add_pool` ile tüm graf tek bir vektöre indirgenir.
* **Loss Function:** `TripletMarginLoss` kullanılarak, benign varyantlar orijinale çekilirken, malignant varyantlar itilir.

### Hiperparametreler
Kod içerisindeki temel ayarlar:
* `BATCH_SIZE = 8`: GPU belleğine göre artırılabilir.
* `EPOCHS = 50`: Modelin veriyi kaç kez göreceği. (Artırılacak)
* `MARGIN = 0.2`: Triplet Loss fonksiyonundaki marj değeri.

## 📊 Beklenen Sonuçlar
Başarılı bir eğitim sonunda modelin; hastalık yapan mutasyonları, zararsız olanlardan embedding uzayında (vektörel düzlemde) net bir şekilde ayırması beklenmektedir.
