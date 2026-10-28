# Discorso per la presentazione — Denoising EuroSAT: AutoEncoder vs VAE

**Durata prevista:** 10–12 minuti
**Nota:** i richiami `[Slide N]` seguono la struttura di `slides.tex`. Puoi leggere il testo così com'è oppure usarlo come scaletta.

---

## [Slide 1] Apertura

Buongiorno a tutti. Oggi vi presento il mio project work sul **denoising di immagini satellitari**. Il problema è semplice da enunciare: un'immagine acquisita da satellite può essere degradata da rumore e da sfocatura, e l'obiettivo è ricostruire l'immagine pulita a partire da quella degradata.

La domanda che ho affrontato è di confronto tra due famiglie di modelli: un **AutoEncoder deterministico** e un **VAE**, cioè un autoencoder variazionale con regolarizzazione sullo spazio latente. Per rispondere ho costruito un protocollo controllato sul dataset EuroSAT, con 18 esperimenti e metriche standard di qualità dell'immagine.

## [Slide 2] Obiettivo

L'idea di partenza è che il denoising sia un problema di **ricostruzione immagine-to-immagine**: la rete riceve l'immagine degradata e deve produrre come output l'immagine pulita, che durante l'addestramento è il target.

La seconda parte dell'obiettivo è il **confronto controllato**: capire se la regolarizzazione del VAE aiuta o penalizza quando il compito è il denoising, e a quali livelli di degrado il denoising è davvero conveniente rispetto a non fare nulla.

## [Slide 3] Dataset

Ho usato **EuroSAT**, nella versione RGB: **27.000 immagini** da **64×64 pixel**, distribuite su **10 classi** di copertura del suolo — colture, foresta, vegetazione, autostrade, aree industriali, pascoli, colture permanenti, aree residenziali, fiumi e laghi.

Ho creato uno **split stratificato 70/15/15** con seed fisso 42: circa **18.900 immagini di training, 4.050 di validation e 4.050 di test**, mantenendo la proporzione delle classi in ogni partizione. La stratificazione è importante perché EuroSAT non è perfettamente bilanciato.

## [Slide 4] Il degrado

Il degrado che studio è di due tipi, combinabili.

Il primo è il **rumore gaussiano additivo**, con deviazione standard σ pari a **0.05, 0.15 e 0.30**, su immagini normalizzate in [0, 1]. Il secondo è un **blur gaussiano** opzionale con σ_b pari a **0.5, 1.0 e 2.0** e kernel di dimensione 5, applicato **prima** del rumore.

In questa figura vedete un esempio reale: a sinistra l'immagine originale, poi la stessa immagine con i tre livelli di rumore. Già a occhio si capisce che σ = 0.05 è un disturbo leggero, mentre σ = 0.30 compromette seriamente i dettagli.

## [Slide 5] Le due architetture

I due modelli condividono la stessa struttura di base. L'**encoder** è composto da due convoluzioni con stride 2, che portano l'immagine da 3×64×64 a una mappa 64×16×16; poi c'è uno **spazio latente di 64 dimensioni** e un **decoder** con due convoluzioni trasposte che riportano l'immagine a 3×64×64, con attivazione sigmoide per stare in [0, 1].

La differenza è nel collo di bottiglia. L'**AutoEncoder** è deterministico: produce direttamente un vettore latente e lo decodifica. Il **VAE**, invece, produce una **media μ e una log-varianza**; durante il training campiona z con il reparameterization trick, mentre in valutazione usa **z = μ**, così le metriche sono deterministiche e confrontabili.

## [Slide 6] Loss e protocollo

Le due loss sono:

- **AE:** MSE tra ricostruzione e immagine pulita.
- **VAE:** MSE più **β per la KL** tra la posteriore q(z|y) e il prior N(0, I), con **β = 1e-4**.

Ho mantenuto lo **stesso spazio latente, 64 dimensioni, per entrambi**, così il confronto è il più possibile equo.

Il protocollo completo prevede **18 esperimenti**: 3 livelli di rumore per 3 livelli di blur per 2 modelli. Ogni esperimento usa **5 epoche**, batch size 128, Adam con learning rate 1e-3, **early stopping** con patience 3 e checkpoint scelto in base alla **MSE di ricostruzione sul validation set**. Il rumore di validation e test è generato con seed fissi, quindi ogni modello vede esattamente lo stesso degrado.

## [Slide 7] Metriche

Le metriche sono quelle standard per il denoising:

- **MSE globale**, calcolata su tutti i pixel;
- **PSNR** in decibel, che è una trasformazione logaritmica dell'MSE;
- **SSIM**, riportata come media e deviazione standard per immagine, perché misura la somiglianza strutturale e non solo l'errore pixel-wise;
- **KL divergence**, solo per il VAE;
- e soprattutto la **baseline rumorosa**: cioè le metriche dell'immagine degradata lasciata così com'è.

Quest'ultimo punto è cruciale: un modello di denoising è utile **solo se batte la baseline**, cioè solo se la ricostruzione è migliore dell'input rumoroso.

## [Slide 8] Panoramica dei risultati

Questa è la panoramica. In generale i modelli ricostruiscono immagini con PSNR **tra circa 23,6 e 24,6 dB**, con SSIM tra **0,58 e 0,63**. Il VAE è leggermente sotto l'AE in quasi tutte le configurazioni, ma le differenze sono piccole, dell'ordine di qualche decimo di decibel.

Il dato interessante non è tanto il valore assoluto, quanto il suo andamento rispetto al livello di rumore: i modelli restano sostanzialmente stabili, mentre la baseline cambia molto. Ed è quello che vediamo nella slide successiva.

## [Slide 9] PSNR al variare del rumore

Qui c'è il risultato centrale dell'esperimento. In grigio tratteggiato vedete la baseline, cioè l'immagine rumorosa non elaborata; in blu l'AE e in arancione il VAE.

La baseline **crolla** al crescere del rumore: passa da circa **25,9 dB a σ = 0.05**, a **16,7 dB a σ = 0.15**, fino a **11,7 dB a σ = 0.30**.

I modelli, invece, restano quasi piatti intorno ai **24 dB**. Questo significa due cose. Primo: la qualità della ricostruzione non dipende molto dall'intensità del rumore, ma dalla **capacità del modello**, che in questa architettura ha un limite intrinseco intorno ai 24 dB. Secondo: il punto di incrocio tra baseline e modelli spiega quando il denoising conviene. A rumore molto basso la baseline è già migliore del modello; a rumore alto il modello è nettamente migliore.

## [Slide 10] Guadagno rispetto alla baseline

Questa slide quantifica lo stesso fenomeno in termini di **guadagno di PSNR** rispetto all'immagine rumorosa.

A **σ = 0.05** il guadagno è **negativo o nullo**: l'AE va da **−1,31 a +0,14 dB** a seconda del blur, il VAE da **−1,54 a −0,44 dB**. In pratica, con rumore leggero, applicare il modello **peggiora** l'immagine, perché il rumore è minore dell'errore di ricostruzione del modello.

A **σ = 0.15** il guadagno è di circa **+7,5 dB** per entrambi i modelli. A **σ = 0.30** arriva a circa **+12 dB**.

Il messaggio è chiaro: **il denoising conviene solo quando il degrado è sufficientemente forte**. Con rumore leggero, il limite è la fedeltà ricostruttiva dell'architettura, non il rumore.

## [Slide 11] AE vs VAE

Questo è il confronto diretto tra i due modelli. Ogni punto è una configurazione di degrado; sopra la diagonale vince il VAE, sotto vince l'AE.

Il risultato è che **l'AE è leggermente migliore in 8 configurazioni su 9**. Le differenze di PSNR tra VAE e AE sono:

- a σ = 0.05: **−0,22, −0,26 e −0,58 dB**;
- a σ = 0.15: **−0,14, −0,17 e −0,04 dB**;
- a σ = 0.30: **−0,04, +0,14 e −0,15 dB**.

L'unica configurazione in cui il VAE vince è σ = 0.30 con blur 1,0, con un vantaggio di soli **+0,14 dB**. Anche l'SSIM segue lo stesso andamento: l'AE sta tra **0,583 e 0,626**, il VAE tra **0,587 e 0,616**.

## [Slide 12] La KL del VAE

Per completezza ho misurato anche la **KL divergence** del VAE, che resta stabile tra **2,50 e 2,72** in tutte le configurazioni.

È un valore contenuto, coerente con un **β molto piccolo, 1e-4**: la regolarizzazione c'è ed è attiva, ma il suo peso è modesto. Questo spiega perché il VAE si comporti quasi come un autoencoder deterministico e perché il costo in fedeltà sia così piccolo.

## [Slide 13] Discussione

Mettendo insieme i risultati, la conclusione principale è che **per il denoising puro l'AE deterministico è leggermente preferibile**: ottimizza direttamente la MSE, che è esattamente la metrica di ricostruzione, e non paga il termine KL.

Il VAE, però, non è da scartare: il suo costo in fedeltà è piccolo e in cambio offre uno **spazio latente regolarizzato**, potenzialmente utile se oltre al denoising servisse anche generazione o analisi dello spazio latente.

Il secondo punto è che **il valore del denoising dipende dal livello di degrado**: con rumore molto basso non conviene, con rumore alto il guadagno è enorme, fino a 12 dB.

## [Slide 14] Limiti

Ci sono diversi limiti da dichiarare onestamente.

Il training è breve: **5 epoche**, su CPU, con un **singolo seed**. L'architettura è volutamente semplice e **senza skip connections**, il che spiega il tetto di circa 24 dB. Ho studiato solo **rumore e blur gaussiani**, non altri tipi di degrado. Infine, le metriche sono **pixel-wise**: non ho usato metriche percettive come LPIPS, né un'analisi per classe.

Questi limiti non invalidano il confronto — che è a parità di condizioni — ma delimitano la portata delle conclusioni.

## [Slide 15] Conclusioni e sviluppi

In conclusione:

1. Il denoising funziona e dà guadagni fino a **+12 dB** rispetto alla baseline, ma **solo con rumore sufficientemente forte**.
2. A parità di budget, **l'AE è leggermente migliore del VAE** sulla ricostruzione; il VAE resta interessante per la qualità dello spazio latente.
3. Il limite principale è la **capacità dell'architettura**, non il rumore.

Gli sviluppi naturali sono: aggiungere **skip connections** o passare a una **U-Net**, aumentare epoche e seed, **condizionare il modello sul livello di rumore**, confrontarsi con metodi dedicati come **DnCNN o FFDNet**, e introdurre **metriche percettive** e un'analisi per classe.

## [Slide 16] Chiusura

Con questo ho concluso. Grazie per l'attenzione: sono disponibile per domande.

---

# Appendice — possibili domande e risposte brevi

**Perché l'AE batte il VAE?**
Perché il compito è ricostruzione fedele e l'AE ottimizza direttamente la MSE. Il VAE aggiunge la KL, che spinge il latente verso il prior e introduce un piccolo compromesso sulla fedeltà. Con β = 1e-4 il compromesso è piccolo, ma sistematico.

**Perché a σ = 0.05 il modello peggiora l'immagine?**
Perché il rumore è più debole dell'errore di ricostruzione dell'architettura. Il modello ha un tetto di circa 24 dB dovuto al collo di bottiglia da 64 dimensioni e all'assenza di skip connections; la baseline a rumore basso è già intorno a 25–26 dB.

**Perché usare la stessa dimensione latente per AE e VAE?**
Per rendere il confronto equo: a parità di capacità del collo di bottiglia, l'unica differenza è la parametrizzazione probabilistica e il termine KL.

**La KL è alta o bassa?**
È intorno a 2,5–2,7, un valore moderato. Con β = 1e-4 il suo contributo alla loss è piccolo; per questo il VAE resta vicino all'AE.

**Il confronto è statisticamente robusto?**
No, è un singolo seed per configurazione. È un limite dichiarato; il passo successivo è ripetere con più seed e riportare media e deviazione standard.

**Perché 5 epoche?**
Per contenere i tempi su CPU. Il protocollo è identico per tutti i modelli, quindi il confronto resta valido, ma i valori assoluti non sono quelli di un training lungo.
