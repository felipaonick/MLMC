# 📸 Visualizzazione dei Filtri nelle Reti Convolutive (CNN)

## 🔍 Obiettivo del modulo

Comprendere cosa *vede* un filtro specifico in un determinato layer di una rete convolutiva preaddestrata.
Lo facciamo ottimizzando un'immagine casuale per massimizzare l’attivazione del filtro scelto.

---

## 🧠 Concetti base

* Le **CNN (Convolutional Neural Networks)** analizzano le immagini in maniera gerarchica.
* I **primi layer** rilevano feature semplici (bordi, texture).
* I **layer profondi** rilevano pattern più complessi e specifici (occhi, becchi, forme complesse).

---

## 🛠️ Tecnica utilizzata: Gradient Ascent (Discesa del gradiente inversa) 🎯

Vogliamo ottimizzare una immagine casuale iniziale in modo che vada ad attivare in modo massimale un determinato filtro in un determinato layer della rete neurale.

Quindi andiamo a modificare l'immagine iniziale per fare emergere nell'immagine le feature che vanno poi ad attivare maggiormente il filtro che abbiaom selzionato.

Alla fine di questo processo otteniamo una rappresentazione visiva di ciò che il filtro cerca di identificare.

Con il **Gradient Ascent** non si vanno ad ottimizzare i parametri (pesi) della rete neurale, dato che nella rete pre-addestrata sono già ottimizzati, ma si va ad ottimizzare i pixel che compongono l'immagine iniziale in maniera da matchare il più possibile a quelli che il filtro selezionato è in grado di riconoscere.

### Fasi del processo di ottimizzazione

1. **Inizializzazione**:
   Partiamo da un'immagine completamente casuale, un'immagine completamente rumorosa con valori random uniformi.

2. **Forward Pass**:
   Passiamo l'immagine nella rete e calcoliamo le feature map.

3. **Target**:
   Selezioniamo un filtro specifico in un determinato layer come obiettivo. Poi misuriamo l'attivazione generata dal filtro. Ovvero, dopo l'operazione di convoluzione otteniamo una feature map che poi sarà data in pasto alla funzione di attivazione del layer, e l'output sarà l'attivazione usata nelle fasi successive. 

4. **Calcolo della perdita (loss)**:
   La "perdita" in questo caso non indica un'errore, ma il valore medio dell'attivazione del filtro. Maggiore è il valore medio (trovato all'interno della feature map che va attraverso la funzione di attivazione), più forte è la risposta del filtro all'immagine.
   Vogliamo massimizzare il valore medio dell’attivazione di quel filtro.

5. **Calcolo del gradiente**:
   Calcoliamo il gradiente della loss (valore medio) rispetto ai pixel dell’immagine iniziale.

6. **Aggiornamento dell'immagine**:
   Aggiorniamo l’immagine aggiungendo i gradienti ai pixel dell'immagine originale con un learning rate elevato.

7. **Iterazioni**:
   Ripetiamo questo processo per un numero fissato di volte (epochs) (es. 30).

8. **Post-processing**:
   L’immagine finale viene processata per essere leggibile visivamente in RGB.

---

## 🧪 Esperimento con ResNet50V2 🏗️

* 📥 Modello preaddestrato su ImageNet
* ❌ Rimuoviamo la parte finale di classificazione
* ✅ Manteniamo solo i layer convolutivi

### Layer selezionati:

* `conv2_block1_1_conv` (vicino all’input)
* `conv5_block3_3_conv` (più profondo)

👉 Creiamo un **Feature Extractor** che restituisce le attivazioni di questi layer.

---

## 🖼️ Funzione `visualize_filter` 🧩

Accetta:

* indice del layer
* indice del filtro

Procedura:

1. Genera immagine di rumore casuale
2. Ottimizza l’immagine per massimizzare l’attivazione del filtro selezionato
3. Dopo 30 iterazioni, dove in ogni iterazione si esegue la ascesa del gradiente, restituisce la loss e l'immagine risultante

---
 min 19:53 

## 🧮 Calcolo gradiente con la funzione gradient_ascent_step() 📈

* Registra tutte le operazioni su tensori

* Serve per calcolare il gradiente della loss rispetto all’immagine

### Funzione gradient_ascent_step()

#### @tf.function

Inanzitutto torviamo un deconratore **@tf.function** ch serve per dire a tensorflow di convertire la funzione in un **grafo computazionale**.

Un Grafo Computazionale è uno stratagemma di ottimizzazione di tensorflow che serve a ridurre i tempi di esecuzione della funzione soprattutto quando tale funzione viene ripetuta molte volte come nel ciclo che abbiamo implementato. 

Grazie al decoratore **@tf.function** la memoria occupata dallo script viene gestita in maniera completamente diversa e la utilizziamo quando vogliamo velocizzare l'esecuzione di funzioni computazionalmente impegnative che vengono ripetute per un certo numero di volte.

#### tf.GradientTape()

La funzione esegue un singolo passo di ottimizzazione per modificare l'immagine in maniera da andare a massimizzare l'attivazione di un filtro specifico. Il concetto chiave quì è l'ascesa del gradiente, che è l'opposto della discesa del gradiente, dato che noi vogliamo massimizzare l'attivazione per far emergere i pattern di un determinato filtro.

Per farlo usiamo il **tf.GradientTape()** che registra le operazioni effettuate su un determinato tensore, in questo caso sull'immagine, in modo da poter calcolare più agevolmente il gradiente rispetto ad essa di una certa funzione di costo (compute_loss() che calcola la perdita dell'immagine rispetto ad un determinato livello e a un determinato filtro di tale livello). 


#### compute_loss()

Fa la **predict()** sull'immagine usando la FE quindi l'immagine passa attraverso la rete attivando tutti i filtri di tutti i livelli presenti nella FE.

Dopodichè andiamo a prendere le attivazioni di un determinato livello per un determinato filtro. Ovvero prendiamo tutti i dati in uscita di un determinato filtro (l'attivazione di tale filtro) di un determinato layer (`filter_activation = activation[level_index][:, :, :, filter_index]`)

Dopodichè andiamo a calcolare la media di tutti i valori di sudetta attivazione del filtro selezionato. 

E questa sarà la nostra Loss. Più è grande la Loss e più sarà ampia la risposta che viene data da tale filtro sull'immagine di input. Dobbiamo massimizzare la Loss in questo caso non minimizzarla. 

Più alto è il valore medio risultato dell'attivazione del filtro e più significa che tale filtro si è attivato quindi cerchiamo i valori medi più alti.

#### Calcolo dei gradienti

Andiamo a calcolare i gradienti rispetto all'immagine. Usiamo il risultato della Loss per calcolare i gradienti rispetto all'immagine.


`grads = tape.gradient(loss, img)`

Poi normalizziamo i gradienti per evitare di avere valori troppo grandi o troppo piccoli (dato che poi potremmo avere degli aggiornamenti che possono essere o troppo forti o troppo deboli) quindi spalmiamo su un intervallo più uniforme tutti i gradienti.

Infine andiamo ad **aggiungere** all'immagine iniziale i gradienti così calcolati moltiplicato per il learning rate.

`img += learning_rate * grads`


* Fatto questo il passo di ascesa del gradiente è terminato e l'immagine iniziale è stata modificata dopo un solo step. Ri-iteriamo questo step per 30 volte ed otteniamo una immagine modificata 30 volte in direzione per l'attivazione massimizzata del filtro in questione. Dunque possiamo visualizzare il filtro, o meglio l'attivazione del filtro, dopo che abbiamo dato in input tale immagine modificata.

---

## 🎨 Post-processing dell’immagine finale

1. Centrare i valori intorno a 0 ➖➕
2. Dividere per la deviazione standard ➗
3. Eliminare rumore ai bordi con un **center crop**
4. Riportare tutto nel range \[0, 1] ➡️ RGB
5. Clippare i valori per avere pixel validi

---

## 🔬 Confronto attivazioni nei layer

| Caratteristica            | Layer vicino all'input       | Layer profondo            |
|---------------------------|-----------------------------|---------------------------|
| **Pattern rilevati**       | Bordi, texture semplici      | Forme complesse, parti di oggetti |
| **Struttura delle feature** | Pattern ripetitivi e regolari | Pattern più astratti e articolati |
| **Generalità**             | Universali per qualsiasi immagine | Specifici per la classificazione |
| **Ruolo nel modello**      | Estraggono caratteristiche di base | Combinano feature per interpretare oggetti |
| **Livello di dettaglio**   | Alto (pixel e bordi distinti) | Basso (strutture più grandi e concettuali) |
| **Utilità**                | Supporto per livelli successivi | Riconoscimento di oggetti e categorie |


* I primi layer sono **generici** e rilevano pattern presenti in tutte le immagini
* I layer profondi sono **specifici** e riconoscono caratteristiche legate al dataset di training

---

## 🧩 Visualizzazione multipla dei filtri (64 filtri)

* Usata una funzione per visualizzare i **primi 64 filtri** di un layer
* Risultato mostrato come una griglia 8x8
* Permette di analizzare l’intero comportamento del layer nel suo complesso

---

## 📌 Conclusioni

✅ I **layer iniziali** sono fondamentali per rilevare feature base
✅ I **layer profondi** permettono alla rete di comporre strutture complesse
🎯 La tecnica della **ascesa del gradiente** ci aiuta a interpretare **visivamente** cosa riconoscono i filtri
🔍 Utilissima per il debugging e la comprensione delle CNN


