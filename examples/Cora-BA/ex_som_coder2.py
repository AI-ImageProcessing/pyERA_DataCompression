import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pyERA.som import Som
from pyERA.utils import ExponentialDecay

from metrics import (
    calculate_mse_and_psnr,
    theoretical_storage_bits,
    original_rgb_storage_bits,
    compression_ratio,
)

from run_length import (
    rle_encode_2d,
    rle_decode_2d,
    save_rle,
    estimate_rle_bytes,
)


# Hilfsfunktionen

def save_image_and_hist(img_array, epoch, output_path):
    pixels = img_array.reshape(-1, 3)

    colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True
    )

    fig = plt.figure(figsize=(18, 6))

    # (1) Rekonstruiertes Bild
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_array)
    ax1.set_title(f"Rekonstruiertes Bild – Epoche {epoch}")
    ax1.axis("off")

    # (2) Histogramm
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(
        range(len(colors)),
        counts,
        color=np.array(colors) / 255.0
    )
    ax2.set_title("Farb-Häufigkeit")
    ax2.set_xlabel("Farbindex")
    ax2.set_ylabel("Pixelanzahl")

    # (3) Farblegende
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")

    legend_text = "\n".join(
        f"{i}: RGB{tuple(col)}" for i, col in enumerate(colors)
    )
    ax3.text(0, 1, legend_text, va="top", family="monospace")
    ax3.set_title("Verwendete Farben")

    plt.tight_layout()

    filename = f"epoch_{epoch:03d}_reconstruction.png"
    plt.savefig(os.path.join(output_path, filename), dpi=150)
    plt.close(fig)


def save_som_weights(som, epoch, output_path):
    weights = np.rint(som.return_weights_matrix()).astype(np.uint8)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(weights)
    plt.title(f"SOM-Gewichte – Epoche {epoch}")
    plt.axis("off")

    filename = f"epoch_{epoch:03d}_som_weights.png"
    plt.savefig(os.path.join(output_path, filename), dpi=150)
    plt.close(fig)


def reconstruct_image(som, img_array):
    rows, cols, _ = img_array.shape
    rec = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            pixel = img_array[r, c]
            bmu = som.return_BMU_index(pixel)
            rec[r, c] = np.rint(som.get_unit_weights(bmu[0], bmu[1])).astype(np.uint8)

    return rec


def coder(som, img_uint8, som_size, output_path, epoch):
    """
    Kodiert jedes Pixel über den BMU auf einen Index.
    """
    rows, cols, _ = img_uint8.shape
    cod_rec = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            pixel = img_uint8[r, c]
            bmu = som.return_BMU_index(pixel)
            code = bmu[0] * som_size + bmu[1]
            cod_rec[r, c] = code

    filename = f"epoch_{epoch:03d}_coded_stream.npy"
    np.save(os.path.join(output_path, filename), cod_rec)

    return cod_rec


def decoder(som, cod_rec, som_size):
    rows, cols = cod_rec.shape
    rec = np.zeros((rows, cols, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            code = cod_rec[r, c]
            i = code // som_size
            j = code % som_size
            rec[r, c] = np.rint(som.get_unit_weights(i, j)).astype(np.uint8)

    return rec


def save_metrics_csv_header(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "learning_rate",
                "radius",
                "training_time_sec",
                "epoch_time_sec",
                "mse",
                "psnr_db",
                "original_bytes_theoretical",
                "quantized_bytes_theoretical",
                "compression_ratio_theoretical",
                "coded_stream_npy_bytes",
                "reconstructed_png_bytes",
                "rle_theoretical_bytes",
                "rle_file_bytes"
            ])


def append_metrics_row(csv_path, row):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_metric_plot(csv_path, output_path):
    epochs = []
    mse_values = []
    psnr_values = []

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # Header überspringen
        for line in f:
            parts = line.strip().split(",")

            epoch = int(parts[0])
            mse = float(parts[5])
            psnr = float(parts[6])

            epochs.append(epoch)
            mse_values.append(mse)
            psnr_values.append(psnr)

    # MSE-Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mse_values, marker="o")
    plt.title("MSE über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mse_over_epochs.png"), dpi=150)
    plt.close()

    # PSNR-Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnr_values, marker="o")
    plt.title("PSNR über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "psnr_over_epochs.png"), dpi=150)
    plt.close()


# Hauptprogramm

def main():
    # Parameter
    som_size = 4              # 4x4 SOM = 16 Farben
    batch_size = 256
    tot_epoch = 5
    image_name = "20180110_155149.bmp"

    input_path = "./input/"
    output_path = "./output/"
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, "metrics.csv")

    # Falls du die CSV-Struktur geändert hast, alte Datei vorher löschen
    save_metrics_csv_header(csv_path)

    # Bild laden
    img_original = Image.open(os.path.join(input_path, image_name)).convert("RGB")
    img_original = img_original.resize((512, 512), Image.NEAREST)

    img = np.asarray(img_original, dtype=np.float32)
    img_uint8 = np.asarray(img_original, dtype=np.uint8)

    rows, cols, _ = img.shape
    num_colors = som_size * som_size

    # SOM initialisieren
    som = Som(
        matrix_size=som_size,
        input_size=3,
        low=0,
        high=255,
        round_values=True
    )

    lr_decay = ExponentialDecay(0.5, decay_step=1, decay_rate=0.8, staircase=True)
    rad_decay = ExponentialDecay(2.0, decay_step=1, decay_rate=0.5, staircase=True)

    # Theoretischer Speicherbedarf
    original_storage = original_rgb_storage_bits(rows, cols)
    quantized_storage = theoretical_storage_bits(rows, cols, num_colors, include_palette=True)
    theoretical_ratio = compression_ratio(
        original_storage["total_bytes"],
        quantized_storage["total_bytes"]
    )

    print("Theoretischer Speicherbedarf:")
    print(f"Original:   {original_storage['total_bytes']:.2f} Bytes")
    print(f"Quantisiert:{quantized_storage['total_bytes']:.2f} Bytes")
    print(f"Kompressionsrate (theoretisch): {theoretical_ratio:.4f}")

    # Training
    for epoch in range(1, tot_epoch + 1):
        epoch_start = time.perf_counter()

        lr = lr_decay.return_decayed_value(epoch)
        rad = rad_decay.return_decayed_value(epoch)

        # Zufalls-Batch ziehen
        batch = []
        for _ in range(batch_size):
            rr = np.random.randint(rows)
            cc = np.random.randint(cols)
            batch.append(img[rr, cc])

        # Training
        training_start = time.perf_counter()
        som.training_batch_step(
            batch,
            learning_rate=lr,
            radius=rad,
            weighted_distance=True
        )
        training_end = time.perf_counter()
        training_time = training_end - training_start

        # Rekonstruktion direkt über BMU
        rec_uint8 = reconstruct_image(som, img)

        # Visualisierung speichern
        save_image_and_hist(rec_uint8, epoch, output_path)
        save_som_weights(som, epoch, output_path)

        # Standard-Rekonstruktionspfad
        recon_path = os.path.join(output_path, f"epoch_{epoch:03d}_decoded_from_stream.png")

        # Coder / Decoder
        cod_rec = coder(som, img_uint8, som_size, output_path, epoch)

        rec_decoded = decoder(som, cod_rec, som_size)
        Image.fromarray(rec_decoded).save(recon_path)

        # RLE
        rle_values, rle_counts, rle_rows, rle_cols = rle_encode_2d(cod_rec)

        rle_path = os.path.join(output_path, f"epoch_{epoch:03d}_coded_stream_rle.npz")
        save_rle(rle_path, rle_values, rle_counts, rle_rows, rle_cols)

        # Prüfen, ob die RLE-Dekodierung korrekt ist
        cod_rec_rle_decoded = rle_decode_2d(rle_values, rle_counts, rle_rows, rle_cols)

        if not np.array_equal(cod_rec, cod_rec_rle_decoded):
            raise ValueError(f"RLE-Dekodierung fehlerhaft in Epoche {epoch}")

        rec_from_rle = decoder(som, cod_rec_rle_decoded, som_size)
        Image.fromarray(rec_from_rle).save(
            os.path.join(output_path, f"epoch_{epoch:03d}_decoded_from_rle_stream.png")
        )

        rle_theoretical_bytes = estimate_rle_bytes(rle_values, rle_counts)
        rle_file_bytes = os.path.getsize(rle_path)

        # Fehlerberechnung
        mse, psnr = calculate_mse_and_psnr(img_uint8, rec_decoded)

        # Dateigrößen
        coded_stream_path = os.path.join(output_path, f"epoch_{epoch:03d}_coded_stream.npy")
        coded_stream_npy_bytes = os.path.getsize(coded_stream_path)
        reconstructed_png_bytes = os.path.getsize(recon_path)

        # Gesamtzeit der Epoche
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        # CSV
        append_metrics_row(csv_path, [
            epoch,
            round(lr, 6),
            round(rad, 6),
            round(training_time, 6),
            round(epoch_time, 6),
            round(mse, 6),
            round(psnr, 6),
            round(original_storage["total_bytes"], 2),
            round(quantized_storage["total_bytes"], 2),
            round(theoretical_ratio, 6),
            coded_stream_npy_bytes,
            reconstructed_png_bytes,
            rle_theoretical_bytes,
            rle_file_bytes
        ])

        print(
            f"Epoche {epoch:03d} | "
            f"Lernrate={lr:.4f} | Radius={rad:.4f} | "
            f"MSE={mse:.4f} | PSNR={psnr:.4f} dB | "
            f"Trainingszeit={training_time:.4f}s | "
            f"Epochengesamtzeit={epoch_time:.4f}s | "
            f"RLE-Bytes={rle_theoretical_bytes} | "
            f"RLE-Datei={rle_file_bytes}"
        )

    # SOM speichern
    som.save(output_path, "som_coder.npz")

    # Plots speichern
    save_metric_plot(csv_path, output_path)

    print(f"\nFertig. Ergebnisse in: {output_path}")


if __name__ == "__main__":
    main()