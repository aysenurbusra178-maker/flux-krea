# FLUX-Krea + VIZORA Local Workflow

Bu entegrasyon FLUX-Krea deposunun mevcut inference kodunu değiştirmez. Yeni `vizora_generate.py` komutu, üretimden önce özel VIZORA sidecar servisinden sürümlü prompt alır ve üretim sonrasında kalite sonucunu aynı kütüphaneye geri gönderir.

## Gerekli ortam değişkenleri

Aynı laptop:

```powershell
$env:VIZORA_SIDECAR_URL = "http://127.0.0.1:4317"
$env:VIZORA_SIDECAR_TOKEN = "same-long-token-used-by-the-vizora-sidecar"
```

Sidecar ikinci laptopta çalışıyorsa:

```powershell
$env:VIZORA_SIDECAR_URL = "http://192.168.1.50:4317"
$env:VIZORA_SIDECAR_TOKEN = "same-long-token-used-by-the-vizora-sidecar"
$env:VIZORA_ALLOW_LAN = "true"
```

Yalnızca güvenilir özel ağ IP adresi kullanılmalıdır. Public sidecar endpoint'i varsayılan olarak reddedilir.

## Prompt kütüphanesiyle üretim

```powershell
python vizora_generate.py generate `
  --objective "Create an ultra-realistic editorial restaurant portrait" `
  --composition "4:5 vertical portrait composition" `
  --camera "professional DSLR, 85mm lens, natural perspective" `
  --lighting "direct restaurant flash with realistic falloff" `
  --output output/portrait.png
```

Komut iki dosya üretir:

```text
output/portrait.png
output/portrait.png.vizora.json
```

İkinci dosya prompt kimliği, hash, aktif modül sürümleri ve kalite kurallarını içeren üretim makbuzudur.

## Otomatik yerel değerlendirme

VIZORA sidecar bir vision-language model endpoint'ine bağlıysa:

```powershell
python vizora_generate.py generate `
  --objective "Create an authentic unretouched fashion portrait" `
  --output output/portrait.png `
  --auto-evaluate `
  --identity-reference input/identity.jpg
```

Kimlik referansı verilirse evaluator kimlik benzerliğini de puanlar. Referans verilmezse kimlik puanı kararın ana unsuru olarak kullanılmamalıdır.

## Manuel puanlama

```powershell
python vizora_generate.py rate `
  --receipt output/portrait.png.vizora.json `
  --accepted `
  --skin 0.90 `
  --anatomy 0.86 `
  --hands-feet 0.80 `
  --hair 0.88 `
  --lighting 0.92 `
  --composition 0.90 `
  --notes "Skin is strong; fingertips need improvement." `
  --output-path output/portrait.png
```

Puanlar `0.0` ile `1.0` arasındadır. Görülemeyen veya değerlendirilemeyen boyut için puan verilmemelidir.

## Aday prompt sürümü üretme

Kütüphanede yeterli geri bildirim varsa:

```powershell
python vizora_generate.py evolve
```

Bu işlem yalnızca candidate dosyası oluşturur. Aktif kütüphane otomatik değişmez.

## Önemli donanım notu

`FLUX.1 Krea [dev]` 12B parametreli ağır bir modeldir. Bu depodaki orijinal inference script'i tüm ana modelleri GPU'ya taşıdığı için 12 GB VRAM üzerinde doğrudan çalışmayabilir. 12 GB sistemde ana üretim yolu olarak ComfyUI quantized/offload workflow'u kullanılmalı; bu bridge ise prompt kütüphanesini FLUX-Krea ile de ortaklaştırmak için korunmalıdır.

## Gizlilik

- Görseller varsayılan olarak üçüncü taraf API'lere gönderilmez.
- Otomatik değerlendirme yalnızca yapılandırılan yerel model endpoint'ine gider.
- Token kaynak koda veya GitHub'a yazılmamalıdır.
- Üretim ve kimlik görselleri repoya commit edilmemelidir.
