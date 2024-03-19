# Grok-1

Bu depo, Grok-1 açık ağırlık modelini yükleme ve çalıştırma için JAX örnek kodunu içerir.

Ağırlık dosyasını indirip `ckpt-0` dizinini `checkpoints` içine yerleştirdiğinizden emin olun - indirme adımları için [Ağırlıkları İndirme](#downloading-the-weights) bölümüne bakın.

Ardından,

```shell
pip install -r requirements.txt
python run.py
```

kodu test etmek için.

Betik, bir test girdisinde modelden örnekler almak için kontrol noktasını yükler.

Modelin büyük boyutu (314B parametre) nedeniyle, örnek kodla modeli test etmek için yeterli GPU belleğine sahip bir makine gereklidir.
Bu depodaki MoE katmanının uygulanması verimli değildir. Modelin doğruluğunu doğrulamak için özel çekirdeklerin gereksinimi önlenmek amacıyla bu uygulama seçilmiştir.

# Model Özellikleri

Grok-1 şu anda aşağıdaki özelliklerle tasarlanmıştır:

- **Parametreler:** 314B
- **Mimarisi:** 8 Uzmanın Karışımı (MoE)
- **Uzmanların Kullanımı:** Her bir belirteç için 2 uzman
- **Katmanlar:** 64
- **Dikkat Kafaları:** Sorgular için 48, anahtarlar/değerler için 8
- **Gömme Boyutu:** 6,144
- **Belirteçleme:** 131,072 belirteçli SentencePiece belirteçleyici
- **Ek Özellikler:**
  - Dönme gömüler (RoPE)
  - Aktivasyon kırılması ve 8 bitlik nicemleme desteği
- **Maksimum Sıra Uzunluğu (bağlam):** 8,192 belirteç


# Ağırlıkları İndirme

Ağırlıkları bir torrent istemcisi kullanarak ve bu mıknatıs bağlantısını kullanarak indirebilirsiniz:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

veya doğrudan kullanarak : [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):

```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

# Lisans

Bu sürümdeki kod ve ilişkili Grok-1 ağırlıkları Apache 2.0 lisansı altındadır. 
Lisans yalnızca bu depodaki kaynak dosyalarına ve Grok-1 modelinin ağırlıklarına uygulanır.
