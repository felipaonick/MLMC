# 🧠 Confronto tra due reti CNN: con e senza Global Pooling

## 📁 Dataset utilizzato: `CatsVsDogs` (TensorFlow Datasets)

* ✅ Usato in modalità `as_supervised`: ogni elemento è una **tupla (immagine, label)**.
* ✅ La label:

  * `1` = 🐶 **Cane**
  * `0` = 🐱 **Gatto**
* 🔢 Numero di esempi: **20.936**

---

## 🧹 Preprocessing immagini

* 🧭 Cast a `float32`
* ⚖️ Normalizzazione pixel tra `0` e `1`
* 📏 Resize intelligente con funzione TensorFlow
* 📦 Divisione in `train_dataset` e `test_dataset`

---

## 🧪 Differenze nella preparazione

### 🔬 Dataset di **test**:

* Solo specifica `batch size` (una sola epoca, semplice)

### 🏋️ Dataset di **train**:

* Shuffle con buffer pari al numero di esempi
* `repeat()` ➕ `shuffle()` ogni epoca
* `batch_size = 64`
* `steps_per_epoch = num_examples // batch_size` ➡️ calcolato dinamicamente
* ⚠️ **Nessun caricamento in RAM iniziale**: utilizzo intelligente di `prefetch`, abbiamo specificato come preprocessare i dataset di train e di validation e di come deve essere fatta la fase di training (dimensione del batch, steps per epoca e rimescolamento delle immagini) ma senza che nessuna immagine sia ancora caricata in memoria. Questo è il vantaggio di usare **tensorflow-datasets** perchè usa un dataset di `prefetch` in maniera intelligente dato che alcuni dataset potebbero essere enormi e non starci nella memoria.

---

## 🧱 Architettura della rete **senza** Global Pooling

### 🔄 Rete `Sequential`:

* 3️⃣ Blocchi convolutivi (Conv2D ➕ BatchNorm ➕ ReLU ➕ MaxPooling)
* 🧯 Flatten
* 🔢 2 livelli `Dense` (il primo con 64 neuroni, il secondo con 1 neurone ➕ `sigmoid`)

### 🧮 Parametri:

* 🔢 Totali: **\~900.000**
* 🧨 \~820.000 solo nel livello `Dense`
* Motivo: `Flatten` produce **12.800** valori (da feature map 10x10x128)

### ⚙️ Addestramento:

* `25 epoch`
* `fit()` con steps predefiniti
* 🚀 Accuracy finale: **99%**
* ❌ Accuracy in fase di test: **84,9%**

---

## Exursus Global Pooling

## 🧠 Cos'è il Global Pooling?

🔹 È un'operazione che **trasforma ogni feature map (H × W)** in **un singolo numero**, **collassando l’informazione spaziale** su tutta la mappa.

Due versioni comuni:

* **Global Average Pooling (GAP)** → fa la media di tutti i valori di ciascuna feature map
* **Global Max Pooling (GMP)** → prende il valore massimo di ciascuna feature map

---

## 📦 Esempio concreto

Supponiamo di avere una feature map di **dimensioni 7×7×128**, cioè:

* 128 feature maps (canali)
* Ognuna di 7×7 pixel

### Dopo `GlobalAveragePooling2D()`:

👉 Ottieni un vettore **1×1×128**

---

## 📍 Cosa succede all'informazione spaziale?

| Aspetto                                                  | Cosa succede con Global Pooling   |
| -------------------------------------------------------- | --------------------------------- |
| **Informazione spaziale locale (posizione, forma)**      | ❌ Persa                           |
| **Presenza o attivazione di pattern** (es. linee, bordi) | ✅ Mantenuta                       |
| **Efficienza computazionale**                            | ✅ Molto alta                      |
| **Numero di parametri**                                  | ✅ Nessuno (è un'operazione fissa) |

---

## 🎯 Quando usarlo?

✅ Quando:

* Vuoi **ridurre drasticamente la dimensionalità** prima di un layer denso
* L’importante è **il “cosa c’è”**, non **“dove” è**
* Usi architetture **senza Flatten + Dense**, più leggere (es. MobileNet)

❌ Evita se:

* Devi **preservare posizione** o **relazioni spaziali** (es. segmentazione, detection)

---

## 🆚 Global Pooling vs Flatten

| Caratteristica           | Global Pooling | Flatten + Dense            |
| ------------------------ | -------------- | -------------------------- |
| Mantiene spatial layout? | ❌ No           | ✅ Sì (tramite connessioni) |
| Parametri?               | 0              | Molti                      |
| Rischio overfitting      | Basso          | Più alto                   |
| Performance              | Più efficiente | Più potente (ma costoso)   |

---

## 🧩 Conclusione

Il **Global Pooling**:

* **Comprima** ogni feature map in un singolo valore
* **Riduce drasticamente la dimensione**
* **Perde la posizione esatta** delle attivazioni
* È molto usato in **modelli leggeri o di classificazione pura**

Se hai bisogno di **preservare la geometria o la forma precisa di un oggetto**, meglio usare `Flatten()` e `Dense`.

---


## 🌀 Rete **con Global Pooling**

### 🧠 Sostituzioni:

* 🔁 Rimosso flatten e i due layer `Dense`
* ✨ Inserito `GlobalMaxPooling2D`
* ➕ Output diretto a `Dense(1, activation='sigmoid')`

### 📉 Parametri:

* 🔢 Totali: **\~94.000**
* 💡 Nessun parametro nel `GlobalMaxPooling` (è un'operazione aritmetica, non apprende)

### 🔍 Cosa fa il `Global Max Pooling`?

* Prende per ogni **feature map (10x10)** il **massimo valore**
* Output finale: **128 valori** (uno per ciascuna feature map)
* Evita il flatten e la moltiplicazione dei parametri ❗

---

## 📊 Confronto delle performance

| 🔧 Architettura              | Parametri totali | Test Accuracy | Loss in test |
| ---------------------------- | ---------------- | ------------- | ------------ |
| Dense (senza Global Pooling) | \~900.000        | 84.9%         | Maggiore     |
| Global Max Pooling           | \~94.000         | 87.9%         | Minore       |

✅ **Global Max Pooling**:

* 🏎️ Addestramento più veloce
* 🧠 Maggiore capacità di **propagare l'errore nei primi layer**
* 🧼 Minor rischio di overfitting grazie alla riduzione dei parametri
* 🧮 Più efficiente in memoria e computazione

---

## 💡 Considerazioni finali

* 📉 Rimuovere i layer `Dense` riduce drasticamente i parametri
* ⚡ La rete risulta più veloce e **generalizza meglio** in fase di test
* 🔁 `Global Max Pooling`:

  * 🧮 Nessun parametro da addestrare
  * 🔬 Seleziona la feature predominante per ogni mappa
* 🚀 La **test accuracy** migliora di 3 punti percentuali!

---
