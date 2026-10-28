# Presentazione — Denoising EuroSAT: AutoEncoder vs VAE

Presentazione Beamer 16:9 (16 slide) per il project work.
Il testo pronunciato è in `discorso.md`; le slide contengono solo i punti chiave.

## File

| File | Descrizione |
| --- | --- |
| `slides.tex` | Sorgente LaTeX della presentazione |
| `slides.pdf` | PDF compilato (16 slide, 16:9) |
| `discorso.md` | Scaletta del discorso (non usata dalla compilazione) |

## Ricompilazione

Dalla cartella `presentation/`:

```bash
pdflatex -interaction=nonstopmode slides.tex
pdflatex -interaction=nonstopmode slides.tex
```

Sono necessarie **due passate**: la prima genera il PDF, la seconda aggiorna il
numero totale di slide nel footer (`\inserttotalframenumber`).

In alternativa, con `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode slides.tex
```

Per pulire i file ausiliari:

```bash
rm -f slides.aux slides.log slides.nav slides.out slides.snm slides.toc slides.fls slides.fdb_latexmk
```

## Requisiti

- TeX Live 2023 con `pdflatex`, `beamer.cls`, `lmodern`, `booktabs`, `amsmath`.
- Pacchetto **pgf** (fornito da `texlive-pictures`): `beamer` lo carica
  obbligatoriamente (`pgfmath`, `pgfcore`, `xxcolor`).
  In questo ambiente `pgf` non era presente nel TeX Live di sistema ed è stato
  installato localmente in `~/texmf/`; se `pdflatex` segnala
  `File 'pgfmath.sty' not found`, installare `texlive-pictures` oppure copiare
  i file `pgf` in `~/texmf/tex/{latex,generic}/pgf/` ed eseguire `mktexlsr ~/texmf`.
- **NON** serve `babel` né `italian.ldf`: il testo è in italiano ma la lingua
  non è caricata (il file `italian.ldf` non è installato). Non aggiungere
  `\usepackage[italian]{babel}`.

## Figure

Le immagini sono lette da `../outputs/report/` tramite:

```latex
\graphicspath{{../outputs/report/}}
```

Quindi la compilazione va eseguita da `presentation/` (o mantenendo la stessa
posizione relativa della cartella `outputs/report/`).
Figure usate: `example_degradation.png`, `psnr_bars.png`, `psnr_vs_noise.png`,
`psnr_gain.png`, `ae_vs_vae_psnr.png`, `kl_bars.png`.

## Struttura delle slide

1. Titolo
2. Obiettivo
3. Dataset EuroSAT
4. Il degrado
5. Le due architetture
6. Loss e protocollo
7. Metriche
8. Panoramica dei risultati
9. PSNR al variare del rumore
10. Guadagno rispetto alla baseline
11. AE vs VAE
12. La KL del VAE
13. Discussione
14. Limiti
15. Conclusioni e sviluppi
16. Grazie

## Personalizzazione rapida

- Colori: `primary`, `accent`, `ink`, `muted` nel preambolo di `slides.tex`.
- Dimensione figure: primo argomento opzionale del comando `\fig`
  (es. `\fig[11cm]{psnr_bars.png}{...}`).
- Footer: template `footline` nel preambolo (autore, titolo breve, pagina).
