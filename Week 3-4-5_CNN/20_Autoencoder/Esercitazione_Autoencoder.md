# 🧠 Autoencoder: Compressione, Denoising e Pulizia di Testi Scannerizzati

## 📌 Obiettivi della Esercitazione

1. 🌀 Comprimere dati MNIST (estraendo feature principali)
2. ❌ Ricostruire immagini a partire da dati **rumorosi**
3. 🧾 Pulire immagini scannerizzate con testo sporco da rumore fisico (es. macchie da bicchieri)

---

## 🔧 Setup Iniziale: Importazione Librerie

* 📚 NumPy, Matplotlib.pyplot
* 📦 Dataset `MNIST` da Keras
* 🧱 Layers: `Conv2D`, `MaxPooling2D`, `UpSampling2D`, `BatchNormalization`
* 🧪 Ottimizzatore: `Nadam`
* 🖼️ Funzioni di preprocessing delle immagini (Keras)

---

## 🗂️ Preparazione dei Dati

### 🎨 Normalizzazione e Reshape

* Immagini reshape in `28x28x1`
* Valori normalizzati su scala `[0, 1]`

### 🌪️ Aggiunta di Rumore

* Rumore aggiunto tramite:

  ```python
  noisy = original + 0.65 * np.random.normal(0, 1, shape)
  ```
* Clipping con `np.clip()` per mantenere i valori tra `0` e `1`

---

## 🧠 Costruzione dell’Autoencoder

### 🔄 Architettura Encoder (con Functional API)

* Input: `(28, 28, 1)`
* 2 blocchi convolutivi con:

  * `Conv2D (3x3) + ReLU + Padding='same'`
  * `MaxPooling2D`
  * `BatchNormalization`
* Output encoder: **rappresentazione latente** `7x7x64`

### 🔁 Architettura Decoder (speculare)

* Uso di:

  * `Conv2D`
  * `UpSampling2D`
  * `BatchNormalization`
* Output finale:

  ```python
  Conv2D(filters=1, kernel_size=3x3, activation='sigmoid', padding='same')
  ```

  che da in output una immagine 28x28

---

## ⚙️ Addestramento del Modello

* 🔁 Epoche: `100`
* 📦 Batch Size: `1024`
* 🧮 Loss Function: `binary_crossentropy`
* 📉 **Loss decrescente** fino a \~50 epoche, poi si appiattisce → suggerisce possibile tuning del **learning rate**

---

## 🔍 Analisi della Rappresentazione Latente

* Estrazione dei **primi 7 layer** (encoding puro)
* Visualizzazione delle feature map con `matplotlib`
* 👀 Ogni pixel → contiene **feature informative** usate per la ricostruzione
* Immagini più “semplici” (es. rapr. latente del numero 1) → feature map più scure
  Immagini complesse (es. rapr. latente del numero 8) → feature map più attive e quindi più chiare

---

## 📸 Fase di Denoising: Risultati

### 🔹 Con Rumore 65%

* Immagini rumorose → difficilmente leggibili
* Output dell’autoencoder:

  * ✨ Ricostruzioni fedeli nel complesso
  * Alcuni errori marginali (es. 5 vs 6 ambigui)

### 🔸 Con Rumore 80%

* Molte immagini risultano illeggibili a occhio nudo 👀
* Autoencoder riesce comunque a:

  * Recuperare struttura generale
  * Preservare molte feature
  * Alcuni errori di classificazione evidenti

---

## 🧾 Pulizia Testi Scannerizzati

### 📂 Dataset OCR

* Directory:

  * `data.ocr/train`
  * `data.ocr/train_cleaned`
  * `data.ocr/test`
* 📦 Caricamento immagini con `glob`

### 🧼 Preprocessing Immagini

* Resize a `(428, 548, 3)`
* Conversione a `float32`, normalizzazione su `[0, 1]`
* Costruzione di `numpy.ndarray` con shape corretta

### 🧪 Dati Puliti vs Dati Sporchi

* Visualizzazione di:

  * 🔸 Immagine sporca (es. macchie caffè ☕)
  * 🔹 Immagine pulita 

---

## ✅ Considerazioni Finali

| Aspetto                     | Osservazione                                 |
| --------------------------- | -------------------------------------------- |
| 🧠 Rappresentazione Latente | Ben strutturata, anche in presenza di rumore |
| 📉 Performance              | Buona perdita, migliorabile oltre 50 epoche  |
| 🖼️ Ricostruzione OCR       | Funziona anche su immagini reali e degradate |
| 🛠️ Architettura            | Encoder/Decoder speculari + normalizzazione  |
| 🔍 Feature Map              | Intuibili da livello di attivazione visivo   |

---
