"""Confronto tra architetture dell'autoencoder, con selezione sul validation set.

Il report AE vs VAE (`eurosat_denoising.report`) confronta due famiglie di
modelli a parita' di architettura. Questo modulo serve invece alla ricerca
della *miglior configurazione dell'autoencoder*: mette a confronto varianti
architetturali (skip connections, piu' canali, latente piu' grande) sulle
stesse condizioni di degrado.

La scelta del vincitore si fa sulla **PSNR di validation**, mai sul test: il
test viene solo riportato, altrimenti la selezione sarebbe una stima
ottimistica delle prestazioni.
"""
from __future__ import annotations
# Importa argparse per la CLI del report architetturale.
import argparse
# Importa json per il riepilogo delle architetture.
import json
# Importa defaultdict per raggruppare i risultati per architettura.
from collections import defaultdict
# Importa Path per gestire i percorsi.
from pathlib import Path

# Importa numpy per posizionare le barre dei grafici.
import numpy as np

# Riusa il caricamento e la validazione dei JSON del report principale.
from .report import arch_of, corruption_label, load_results, validate_results

# Architettura di riferimento per i confronti (delta rispetto all'AE originale).
REFERENCE_ARCH = "ae"


def aggregate_by_arch(results):
    """Raggruppa i risultati per architettura e ne calcola le metriche medie.

    Le medie sono calcolate sulle configurazioni di degrado presenti per ogni
    architettura; se le configurazioni non coincidono il confronto e' segnalato.
    """
    by_arch = defaultdict(list)
    for record in results:
        by_arch[arch_of(record)].append(record)
    summary = []
    for arch, records in by_arch.items():
        # Scarta i record privi di metriche di validation: senza di esse la
        # selezione non sarebbe possibile secondo il protocollo.
        with_val = [record for record in records if "val_psnr" in record]
        if not with_val:
            continue
        summary.append({
            "arch": arch,
            "n_params": with_val[0].get("n_params"),
            "n_configs": len(with_val),
            "val_psnr": float(np.mean([record["val_psnr"] for record in with_val])),
            "val_ssim": float(np.mean([record["val_ssim_mean"] for record in with_val])),
            "psnr": float(np.mean([record["psnr"] for record in with_val])),
            "ssim_mean": float(np.mean([record["ssim_mean"] for record in with_val])),
            "psnr_gain": float(np.mean([record["psnr"] - record["baseline_psnr"] for record in with_val])),
            "configs": sorted(
                (record["noise_std"], record["blur_sigma"]) for record in with_val
            ),
        })
    # Ordina per PSNR di validation decrescente: il primo e' il candidato migliore.
    summary.sort(key=lambda entry: entry["val_psnr"], reverse=True)
    return summary


def selection_issues(summary):
    """Segnala confronti non omogenei (configurazioni o protocolli diversi)."""
    issues = []
    if not summary:
        return ["Nessun risultato con metriche di validation: rieseguire lo sweep con la versione aggiornata."]
    config_sets = {tuple(entry["configs"]) for entry in summary}
    if len(config_sets) > 1:
        issues.append("architetture valutate su configurazioni diverse: il confronto non e' omogeneo")
    if len(summary) < 2:
        issues.append("serve almeno un'architettura di confronto oltre al riferimento")
    return issues


def format_arch_table(summary):
    """Tabella testuale delle architetture ordinate per PSNR di validation."""
    header = f"{'arch':>14}  {'params':>11}  {'n':>2}  {'valPSNR':>8}  {'testPSNR':>9}  {'dPSNR':>7}  {'SSIM':>6}  {'gain':>7}"
    lines = [header, "-" * len(header)]
    reference = next((entry for entry in summary if entry["arch"] == REFERENCE_ARCH), None)
    for entry in summary:
        params = "-" if entry["n_params"] is None else f"{entry['n_params']:,}"
        delta = "-" if reference is None else f"{entry['psnr'] - reference['psnr']:+.2f}"
        lines.append(
            f"{entry['arch']:>14}  {params:>11}  {entry['n_configs']:>2}  "
            f"{entry['val_psnr']:>8.2f}  {entry['psnr']:>9.2f}  {delta:>7}  "
            f"{entry['ssim_mean']:>6.3f}  {entry['psnr_gain']:>+7.2f}"
        )
    return "\n".join(lines)


def plot_arch_comparison(summary, save_path=None):
    """Barre affiancate di PSNR di validation e di test per architettura."""
    if not summary:
        return None
    # Import locale di matplotlib per non fissare il backend nei notebook.
    import matplotlib.pyplot as plt
    archs = [entry["arch"] for entry in summary]
    positions = np.arange(len(archs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.0, 1.5 * len(archs)), 4.2))
    ax.bar(positions - width / 2, [entry["val_psnr"] for entry in summary], width, label="validation (selezione)")
    ax.bar(positions + width / 2, [entry["psnr"] for entry in summary], width, label="test (riportato)")
    # Etichette con il numero di parametri, utile per giudicare il costo.
    labels = [entry["arch"] if entry["n_params"] is None else f"{entry['arch']}\n{entry['n_params'] / 1e6:.2f} M" for entry in summary]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(min(entry["psnr"] for entry in summary) - 1.5, max(entry["val_psnr"] for entry in summary) + 0.6)
    ax.set_title("Confronto architetture: PSNR di validation e test")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_arch_by_condition(results, save_path=None):
    """PSNR di test per architettura, separato per configurazione di degrado."""
    configs = sorted({(record["noise_std"], record["blur_sigma"]) for record in results if "val_psnr" in record})
    archs = sorted({arch_of(record) for record in results if "val_psnr" in record})
    if not configs or not archs:
        return None
    # Import locale di matplotlib per non fissare il backend nei notebook.
    import matplotlib.pyplot as plt
    positions = np.arange(len(configs))
    width = 0.8 / len(archs)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.7 * len(configs)), 4.2))
    for index, arch in enumerate(archs):
        values = []
        for noise_std, blur_sigma in configs:
            match = [
                record for record in results
                if arch_of(record) == arch and record["noise_std"] == noise_std and record["blur_sigma"] == blur_sigma
            ]
            values.append(float("nan") if not match else match[0]["psnr"])
        ax.bar(positions + (index - (len(archs) - 1) / 2) * width, values, width, label=arch)
    # Baseline rumorosa come riferimento per ogni configurazione.
    baselines = []
    for noise_std, blur_sigma in configs:
        match = [record for record in results if record["noise_std"] == noise_std and record["blur_sigma"] == blur_sigma]
        baselines.append(float("nan") if not match else match[0]["baseline_psnr"])
    ax.scatter(positions, baselines, marker="_", s=300, color="black", label="baseline rumorosa", zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([corruption_label(*config) for config in configs], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR per configurazione di degrado")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def main(argv=None):
    """CLI: confronta le architetture, sceglie la migliore su validation e salva il report."""
    # Backend non interattivo per l'esecuzione da riga di comando.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Confronto architetture AE con selezione su validation")
    parser.add_argument("--sweep-dir", default="outputs/sweep_arch", help="cartella con i JSON delle architetture")
    parser.add_argument("--output-dir", default="outputs/report_arch", help="cartella di destinazione del report")
    args = parser.parse_args(argv)

    results = load_results(args.sweep_dir)
    if not results:
        print(f"Nessun risultato in {args.sweep_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Controlli di coerenza dei JSON (riuso del report principale).
    issues = validate_results(results)
    summary = aggregate_by_arch(results)
    issues += selection_issues(summary)
    (output_dir / "arch_validation.txt").write_text(
        ("\n".join(issues) + "\n") if issues else "OK: nessun problema rilevato.\n"
    )

    (output_dir / "arch_summary.json").write_text(json.dumps(summary, indent=2))
    figures = [
        plot_arch_comparison(summary, output_dir / "arch_psnr.png"),
        plot_arch_by_condition(results, output_dir / "arch_psnr_by_condition.png"),
    ]
    for figure in figures:
        if figure is not None:
            plt.close(figure)

    print(format_arch_table(summary))
    print()
    if summary:
        # La scelta si basa esclusivamente sulla PSNR di validation.
        best = summary[0]
        print(f"Migliore architettura su validation: {best['arch']} (val PSNR {best['val_psnr']:.2f} dB)")
        print(f"Prestazioni di test corrispondenti: PSNR {best['psnr']:.2f} dB, gain {best['psnr_gain']:+.2f} dB")
    if issues:
        print("\nProblemi rilevati:")
        for issue in issues:
            print(f" - {issue}")
    else:
        print("\nControllo automatico: nessun problema rilevato.")
    print(f"\nReport salvato in {output_dir} (arch_summary.json, arch_validation.txt, *.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
