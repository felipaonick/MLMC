# 🧠 Analisi delle CNN tramite Feature Extractor (VGG16)

## 🎯 Obiettivo della Lezione
Utilizzare un **Feature Extractor** su una rete neurale convoluzionale pre-addestrata (**VGG16 con ImageNet**) per:
- Visualizzare e confrontare le **Feature Maps** generate da diversi layer.
- Comprendere **come la rete tratta immagini simili** (es. due gatti) e **diverse** (es. volto umano).
- Analizzare **la progressiva astrazione delle feature** nei layer profondi.

---

## 📚 Architettura VGG16
- Composta da **blocchi convoluzionali + max-pooling**, terminando con **layer flatten e dense**.
- Progressiva **riduzione dimensionale** (da 28×28 → 14×14 → 7×7).
- Progressiva **aumento dei canali** (da 64 a 512).
- Totale: **138 milioni di parametri**.
- Il modello viene caricato **senza la parte finale di classificazione**.

---

## 🧰 Costruzione del Feature Extractor
- Viene creato un `FeatureExtractor` che mantiene gli stessi input del modello originale e **come output tutti i layer intermedi**.

- Creiamo una funzione di utility che permette di:
  - **Visualizzare le attivazioni** (Feature Maps) di layer selezionati.
  - Esportare i risultati sotto forma di immagine (es. `visualizations/activation_layer_<idx>_<imgname>`).

---

## 🖼️ Dataset di Immagini
Tre immagini scelte da **Pexels**:
1. Primo gatto
2. Secondo gatto (simile ma differente)
3. Volto di una ragazza

Ogni immagine viene:
- Ridimensionata alle dimensioni standard di ImageNet.
- Convertita in array NumPy.
- Inserita in batch per passare attraverso il Feature Extractor.

---

## 🔍 Analisi delle Feature Maps

### 🔹 Layer 2 – Primo Gatto
- Le **prime 16 feature maps** mostrano:
  - Una chiarissima presenza del soggetto (il gatto).
  - Alcune attivazioni selettive:
    - Un filtro evidenzia bene il **naso** e una **linea obliqua**.
    - Altri filtri evidenziano **pelo**, **occhi**, **baffi**, e **contorni**.

### 🔹 Layer 2 – Secondo Gatto
- Confrontando le feature maps con il primo:
  - Il filtro che rileva gli **occhi** è attivo anche qui.
  - Altro filtro attiva una regione simile sul **naso destro**.

> ⚠️ Le similitudini tra le attivazioni permettono di inferire che alcune feature siano condivise, mentre altre no.

### 🔹 Layer 2 – Volto della Ragazza
- **Attivazioni (feature maps) differenti** rispetto ai gatti:
  - Alcuni filtri si attivano in corrispondenza degli **occhi**.
  - Altri attivano i contorni dei **capelli**.
  - La rete neurale preaddestrata tratta in maniera completamnetente diversa l'immagine del volto della ragazza rispetto a quelle dei gatti. 
  - Comportamento completamente diverso da quello osservato per i gatti.

---

## 🧠 Layer Profondo (es. Layer 9)
- Gatto:
  - Le feature maps iniziano a perdere l’informazione visiva diretta (il gatto è quasi “scomparso”).
  - Alcune attivazioni **molto localizzate** (es. occhi).

- Ragazza:
  - Alcuni filtri **attivi in corrispondenza dei capelli e occhi**.

---

## 🧪 Analisi Finale – Layer 17
- Gatti:
  - Alcuni filtri risultano **inattivi o identici** (danno la feature map completamente nera) → suggerisce che **non reagiscono** alla presenza dei gatti.

- Ragazza:
  - Alcuni **filtri si attivano chiaramente** (danno la feature map con parti bianche che rilevano caratteristiche), suggerendo che sono specializzati su **caratteristiche umane**.
  - Altri filtri completamente **disattivati**: probabilmente non rilevano caratteristiche umane → specializzati su oggetti/animali.

---

## 💡 Conclusioni
- L’uso sistematico del **Feature Extractor** permette di:
  - **Visualizzare come le reti neurali vedono** un'immagine.
  - Capire **quali filtri si attivano** e **per quali classi di oggetti**.
  - Fornire insight utili sia per la **debugging**, che per la **ricerca scientifica**.

- Potenziale applicativo:
  - Analisi su **grandi moli di immagini**
  - Estensione su **video** per capire come evolve l’attivazione nel tempo

> ✅ Strumento fondamentale per capire e visualizzare cosa accade dentro una CNN!

