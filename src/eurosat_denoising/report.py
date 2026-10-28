"""Report e controllo automatico dei risultati dello sweep AE vs VAE."""
from __future__ import annotations
# Importa argparse per la CLI del report.
import argparse
# Importa csv per esportare la tabella degli esperimenti.
import csv
# Importa json per leggere i risultati e scrivere il riepilogo.
import json
# Importa math per i controlli di validita' dei numeri.
import math
# Importa defaultdict per raggruppare gli esperimenti.
from collections import defaultdict
# Importa Path per gestire i percorsi.
from pathlib import Path

# Importa numpy per posizionare i grafici a barre.
import numpy as np

# Importa il nome file canonico degli esperimenti per il controllo di coerenza.
from .sweep import experiment_filename

# Campi minimi che ogni file di esperimento deve contenere.
REQUIRED_FIELDS = (
    "model", "noise_std", "blur_sigma", "epochs", "beta", "seed", "batch_size",
    "mse", "psnr", "ssim_mean", "ssim_std", "kl", "baseline_mse", "baseline_psnr",
)
# Campi che identificano univocamente una configurazione sperimentale.
CONFIG_FIELDS = ("model", "noise_std", "blur_sigma", "epochs", "beta", "seed")
# Modelli confrontati dal report.
MODELS = ("ae", "vae")
# Etichette leggibili delle metriche, usate nei grafici.
METRIC_LABELS = {"mse": "MSE", "psnr": "PSNR (dB)", "ssim_mean": "SSIM", "kl": "KL"}


def load_results(sweep_dir):
    """Carica tutti i JSON di esperimento presenti in ``sweep_dir``."""
    sweep_dir = Path(sweep_dir)
    results = []
    for path in sorted(sweep_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON non valido in {path}: {exc}") from exc
        # Conserva il nome del file per i controlli di coerenza con i parametri.
        record["_source"] = path.name
        results.append(record)
    return results


def corruption_label(noise_std, blur_sigma):
    """Etichetta leggibile di una configurazione di degrado."""
    blur_part = "no blur" if blur_sigma is None else f"blur {blur_sigma:g}"
    return f"noise {noise_std:g}\n{blur_part}"


def validate_results(results):
    """Controlla completezza, coerenza e confrontabilita' dei risultati dello sweep."""
    issues = []
    if not results:
        return ["Nessun risultato trovato: eseguire prima lo sweep."]
    seen = {}
    for record in results:
        source = record.get("_source", "<sconosciuto>")
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            issues.append(f"{source}: campi mancanti: {', '.join(missing)}")
            continue
        # Il nome del file deve corrispondere ai parametri salvati al suo interno.
        expected = experiment_filename(record)
        if expected != source:
            issues.append(f"{source}: nome file non coerente con i parametri (atteso: {expected})")
        key = tuple(record[field] for field in CONFIG_FIELDS)
        if key in seen:
            issues.append(f"{source}: configurazione duplicata di {seen[key]}")
        seen[key] = source
        if record["model"] not in MODELS:
            issues.append(f"{source}: modello sconosciuto {record['model']!r}")
        if not (isinstance(record["mse"], (int, float)) and record["mse"] > 0):
            issues.append(f"{source}: MSE non valido {record['mse']!r}")
        if not math.isfinite(record["psnr"]):
            issues.append(f"{source}: PSNR non finito {record['psnr']!r}")
        if not (-1.0 <= record["ssim_mean"] <= 1.0):
            issues.append(f"{source}: SSIM fuori dall'intervallo [-1, 1]: {record['ssim_mean']!r}")
    # La baseline dipende solo dal degrado e dallo split: deve essere identica tra i modelli.
    groups = defaultdict(list)
    for record in results:
        if all(field in record for field in ("noise_std", "blur_sigma", "baseline_mse", "baseline_psnr")):
            groups[(record["noise_std"], record["blur_sigma"])].append(record)
    for (noise_std, blur_sigma), group in groups.items():
        if len({round(r["baseline_mse"], 12) for r in group}) > 1 or len({round(r["baseline_psnr"], 9) for r in group}) > 1:
            issues.append(f"baseline incoerente per noise={noise_std:g}, blur={blur_sigma}: possibili split o seed diversi")
        present = {record["model"] for record in group}
        if present != set(MODELS):
            issues.append(f"confronto incompleto per noise={noise_std:g}, blur={blur_sigma}: modelli presenti {sorted(present)}")
    # Tutti gli esperimenti devono usare lo stesso protocollo per un confronto equo.
    protocols = {(record.get("epochs"), record.get("beta"), record.get("seed")) for record in results}
    if len(protocols) > 1:
        issues.append(f"protocolli misti (epochs/beta/seed): {sorted(protocols, key=str)}")
    return issues


def experiment_table(results):
    """Costruisce le righe della tabella riassuntiva degli esperimenti."""
    rows = []
    for record in sorted(results, key=lambda r: (r["noise_std"], r["blur_sigma"] is not None, r["blur_sigma"] or 0.0, r["model"])):
        baseline_psnr = record.get("baseline_psnr")
        rows.append({
            "model": record["model"],
            "noise_std": record["noise_std"],
            "blur_sigma": record["blur_sigma"],
            "epochs": record["epochs"],
            "beta": record["beta"],
            "mse": record["mse"],
            "psnr": record["psnr"],
            "ssim_mean": record["ssim_mean"],
            "ssim_std": record["ssim_std"],
            "kl": record.get("kl"),
            "baseline_mse": record.get("baseline_mse"),
            "baseline_psnr": baseline_psnr,
            "psnr_gain": None if baseline_psnr is None else record["psnr"] - baseline_psnr,
        })
    return rows


# Colonne della tabella testuale: nome interno, intestazione, larghezza, formato.
_TABLE_FIELDS = (
    ("model", "mod", 4, "{}"),
    ("noise_std", "noise", 5, "{:g}"),
    ("blur_sigma", "blur", 5, "{:g}"),
    ("epochs", "ep", 3, "{}"),
    ("beta", "beta", 7, "{:g}"),
    ("mse", "mse", 9, "{:.6f}"),
    ("psnr", "psnr", 7, "{:.2f}"),
    ("ssim_mean", "ssim", 7, "{:.4f}"),
    ("kl", "kl", 7, "{:.4f}"),
    ("psnr_gain", "gain", 7, "{:+.2f}"),
)


def _format_value(value, fmt, width):
    """Formatta un valore per la tabella testuale, gestendo i valori assenti."""
    text = "-" if value is None else fmt.format(value)
    return f"{text:>{width}}"


def format_table(rows):
    """Formatta la tabella degli esperimenti come testo allineato."""
    header = "  ".join(f"{name:>{width}}" for _, name, width, _ in _TABLE_FIELDS)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join(_format_value(row.get(field), fmt, width) for field, _, width, fmt in _TABLE_FIELDS))
    return "\n".join(lines)


def summarize_comparisons(results):
    """Confronta AE e VAE per ogni configurazione di degrado."""
    groups = defaultdict(dict)
    for record in results:
        groups[(record["noise_std"], record["blur_sigma"])][record["model"]] = record
    summary = []
    for (noise_std, blur_sigma), by_model in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1] is not None, item[0][1] or 0.0)):
        ae, vae = by_model.get("ae"), by_model.get("vae")
        entry = {"noise_std": noise_std, "blur_sigma": blur_sigma}
        if ae is not None and vae is not None:
            entry.update({
                "delta_psnr": vae["psnr"] - ae["psnr"],
                "delta_mse": vae["mse"] - ae["mse"],
                "delta_ssim": vae["ssim_mean"] - ae["ssim_mean"],
                "gain_ae": ae["psnr"] - ae["baseline_psnr"],
                "gain_vae": vae["psnr"] - vae["baseline_psnr"],
                "kl_vae": vae.get("kl"),
            })
        summary.append(entry)
    return summary


def format_summary(summary):
    """Formatta il confronto AE vs VAE come tabella testuale."""
    header = f"{'noise':>5}  {'blur':>5}  {'dPSNR':>7}  {'dMSE':>9}  {'dSSIM':>7}  {'gainAE':>7}  {'gainVAE':>8}  {'klVAE':>8}"
    lines = [header, "-" * len(header)]
    for entry in summary:
        lines.append("  ".join((
            _format_value(entry.get("noise_std"), "{:g}", 5),
            _format_value(entry.get("blur_sigma"), "{:g}", 5),
            _format_value(entry.get("delta_psnr"), "{:+.2f}", 7),
            _format_value(entry.get("delta_mse"), "{:+.6f}", 9),
            _format_value(entry.get("delta_ssim"), "{:+.4f}", 7),
            _format_value(entry.get("gain_ae"), "{:+.2f}", 7),
            _format_value(entry.get("gain_vae"), "{:+.2f}", 8),
            _format_value(entry.get("kl_vae"), "{:.4f}", 8),
        )))
    return "\n".join(lines)


def write_csv(rows, path):
    """Salva la tabella degli esperimenti in formato CSV."""
    path = Path(path)
    if not rows:
        path.write_text("")
        return path
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _configs(results):
    """Elenco ordinato delle configurazioni (noise_std, blur_sigma)."""
    keys = {(record["noise_std"], record["blur_sigma"]) for record in results}
    return sorted(keys, key=lambda key: (key[0], key[1] is not None, key[1] or 0.0))


def _lookup(results, noise_std, blur_sigma, model):
    """Cerca il risultato di un modello per una specifica configurazione di degrado."""
    for record in results:
        if record["noise_std"] == noise_std and record["blur_sigma"] == blur_sigma and record["model"] == model:
            return record
    return None


def plot_metric_bars(results, metric, save_path=None, ylabel=None, title=None):
    """Grafico a barre della metrica per configurazione, AE vs VAE."""
    configs = _configs(results)
    if not configs:
        return None
    # Import locale di matplotlib per non fissare il backend nei notebook.
    import matplotlib.pyplot as plt
    labels = [corruption_label(*key) for key in configs]
    positions = np.arange(len(configs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(configs)), 4.2))
    for offset, model in ((-width / 2, "ae"), (width / 2, "vae")):
        values = []
        for key in configs:
            record = _lookup(results, key[0], key[1], model)
            values.append(float("nan") if record is None else record[metric])
        ax.bar(positions + offset, values, width, label=model.upper())
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    # Etichetta leggibile della metrica (PSNR/MSE/SSIM/KL).
    label = METRIC_LABELS.get(metric, metric)
    ax.set_ylabel(ylabel or label)
    ax.set_title(title or f"{label} per configurazione di degrado")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metric_vs_noise(results, metric, save_path=None):
    """Andamento della metrica al variare del rumore, un pannello per livello di blur."""
    noise_values = sorted({record["noise_std"] for record in results})
    blur_values = sorted({record["blur_sigma"] for record in results}, key=lambda blur: (blur is not None, blur or 0.0))
    # Servono almeno due livelli di rumore per tracciare una curva.
    if len(noise_values) < 2 or not blur_values:
        return None
    # Import locale di matplotlib per non fissare il backend nei notebook.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(blur_values), figsize=(4.6 * len(blur_values), 4.0), squeeze=False)
    for ax, blur_sigma in zip(axes[0], blur_values):
        for model in MODELS:
            xs, ys = [], []
            for noise_std in noise_values:
                record = _lookup(results, noise_std, blur_sigma, model)
                if record is not None:
                    xs.append(noise_std)
                    ys.append(record[metric])
            if xs:
                ax.plot(xs, ys, marker="o", label=model.upper())
        # Baseline rumorosa (identica per AE e VAE) come riferimento tratteggiato.
        baseline_field = {"psnr": "baseline_psnr", "mse": "baseline_mse"}.get(metric)
        if baseline_field is not None:
            xs, ys = [], []
            for noise_std in noise_values:
                record = _lookup(results, noise_std, blur_sigma, "ae") or _lookup(results, noise_std, blur_sigma, "vae")
                if record is not None:
                    xs.append(noise_std)
                    ys.append(record[baseline_field])
            if xs:
                ax.plot(xs, ys, linestyle="--", color="gray", label="baseline")
        ax.set_title("no blur" if blur_sigma is None else f"blur {blur_sigma:g}")
        ax.set_xlabel("noise std")
        label = METRIC_LABELS.get(metric, metric)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"{METRIC_LABELS.get(metric, metric)} al variare del rumore")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_baseline_gain(results, save_path=None):
    """Guadagno di PSNR rispetto alla baseline rumorosa, per configurazione e modello."""
    configs = _configs(results)
    if not configs:
        return None
    # Import locale di matplotlib per non fissare il backend nei notebook.
    import matplotlib.pyplot as plt
    labels = [corruption_label(*key) for key in configs]
    positions = np.arange(len(configs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(configs)), 4.2))
    for offset, model in ((-width / 2, "ae"), (width / 2, "vae")):
        gains = []
        for key in configs:
            record = _lookup(results, key[0], key[1], model)
            gains.append(float("nan") if record is None or record.get("baseline_psnr") is None else record["psnr"] - record["baseline_psnr"])
        ax.bar(positions + offset, gains, width, label=model.upper())
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PSNR gain vs baseline (dB)")
    ax.set_title("Miglioramento rispetto all'immagine rumorosa")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ae_vs_vae_scatter(results, metric="psnr", save_path=None):
    """Scatter AE vs VAE: per PSNR/SSIM sopra la diagonale vince il VAE."""
    pairs = []
    for key in _configs(results):
        ae = _lookup(results, key[0], key[1], "ae")
        vae = _lookup(results, key[0], key[1], "vae")
        if ae is not None and vae is not None:
            pairs.append((ae[metric], vae[metric]))
    if not pairs:
        return None
    # Import locale di matplotlib per non fissare il backend nei notebook.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    ax.scatter(xs, ys, s=60)
    low = min(min(xs), min(ys))
    high = max(max(xs), max(ys))
    pad = 0.05 * (high - low or 1.0)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color="gray", label="parita'")
    label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(f"{label} AE")
    ax.set_ylabel(f"{label} VAE")
    ax.set_title(f"Confronto diretto {label}: AE vs VAE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def main(argv=None):
    """CLI del report: valida lo sweep, stampa le tabelle e salva CSV e grafici."""
    # Backend non interattivo per l'esecuzione da riga di comando.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Report automatico dello sweep AE vs VAE")
    parser.add_argument("--sweep-dir", default="outputs/sweep", help="cartella con i JSON dello sweep")
    parser.add_argument("--output-dir", default="outputs/report", help="cartella di destinazione del report")
    args = parser.parse_args(argv)

    results = load_results(args.sweep_dir)
    if not results:
        print(f"Nessun risultato in {args.sweep_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Controllo automatico: problemi di coerenza e confrontabilita'.
    issues = validate_results(results)
    (output_dir / "validation.txt").write_text(
        ("\n".join(issues) + "\n") if issues else "OK: nessun problema rilevato.\n"
    )

    rows = experiment_table(results)
    write_csv(rows, output_dir / "results.csv")
    summary = summarize_comparisons(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Grafici principali (obbligatori) e opzionali (dipendono dai dati disponibili).
    figures = [
        plot_metric_bars(results, "psnr", output_dir / "psnr_bars.png", ylabel="PSNR (dB)"),
        plot_metric_bars(results, "mse", output_dir / "mse_bars.png", ylabel="MSE"),
        plot_metric_bars(results, "ssim_mean", output_dir / "ssim_bars.png", ylabel="SSIM"),
        plot_baseline_gain(results, output_dir / "psnr_gain.png"),
    ]
    for optional in (
        plot_metric_bars([record for record in results if record["model"] == "vae"], "kl", output_dir / "kl_bars.png", ylabel="KL", title="KL divergence (solo VAE)"),
        plot_metric_vs_noise(results, "psnr", output_dir / "psnr_vs_noise.png"),
        plot_ae_vs_vae_scatter(results, "psnr", output_dir / "ae_vs_vae_psnr.png"),
    ):
        if optional is not None:
            figures.append(optional)
    for figure in figures:
        plt.close(figure)

    print(format_table(rows))
    print()
    print(format_summary(summary))
    print()
    if issues:
        print("Problemi rilevati:")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("Controllo automatico: nessun problema rilevato.")
    print(f"\nReport salvato in {output_dir} (results.csv, summary.json, validation.txt, *.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
