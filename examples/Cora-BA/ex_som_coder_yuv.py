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
    original_rgb_storage_bits,
    compression_ratio,
)

from run_length import (
    rle_encode_2d,
    rle_decode_2d,
    save_rle,
    estimate_rle_bytes,
)

from color_transform import (
    rgb_to_yuv,
    quantize_yuv,
    dequantize_yuv,
    yuv_to_rgb,
    yuv_q_to_som_input,
    som_output_to_yuv_q,
)


def yuv_som_storage_bits(rows, cols, num_colors, y_bits=6, u_bits=4, v_bits=4):
    bits_per_index = int(np.ceil(np.log2(num_colors)))
    index_bits = rows * cols * bits_per_index

    palette_bits_per_color = y_bits + u_bits + v_bits
    palette_bits = num_colors * palette_bits_per_color

    total_bits = index_bits + palette_bits

    return {
        "bits_per_index": bits_per_index,
        "index_bits": index_bits,
        "palette_bits": palette_bits,
        "total_bits": total_bits,
        "total_bytes": total_bits / 8.0,
    }


def yuv_644_storage_bits(rows, cols):
    total_bits = rows * cols * 14
    return {
        "total_bits": total_bits,
        "total_bytes": total_bits / 8.0,
    }


def save_image_and_hist(img_array, epoch, output_path):
    pixels = img_array.reshape(-1, 3)

    colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True
    )

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_array)
    ax1.set_title(f"Rekonstruiertes Bild – Epoche {epoch}")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.bar(
        range(len(colors)),
        counts,
        color=np.array(colors) / 255.0
    )
    ax2.set_title("Farb-Häufigkeit")
    ax2.set_xlabel("Farbindex")
    ax2.set_ylabel("Pixelanzahl")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis("off")

    legend_text = "\n".join(
        f"{i}: RGB{tuple(col)}" for i, col in enumerate(colors)
    )
    ax3.text(0, 1, legend_text, va="top", family="monospace")
    ax3.set_title("Verwendete Farben")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"epoch_{epoch:03d}_reconstruction.png"), dpi=150)
    plt.close(fig)


def save_som_weights_yuv_as_rgb(som, epoch, output_path):
    weights = som.return_weights_matrix()

    yuv_q = som_output_to_yuv_q(weights, y_bits=6, u_bits=4, v_bits=4)
    yuv = dequantize_yuv(yuv_q, y_bits=6, u_bits=4, v_bits=4)
    rgb_weights = yuv_to_rgb(yuv)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(rgb_weights)
    plt.title(f"SOM-Gewichte YUV→RGB – Epoche {epoch}")
    plt.axis("off")

    plt.savefig(os.path.join(output_path, f"epoch_{epoch:03d}_som_weights.png"), dpi=150)
    plt.close(fig)


def coder(som, som_input, som_size, output_path, epoch):
    rows, cols, _ = som_input.shape
    cod_rec = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            pixel = som_input[r, c]
            bmu = som.return_BMU_index(pixel)
            code = bmu[0] * som_size + bmu[1]
            cod_rec[r, c] = code

    np.save(os.path.join(output_path, f"epoch_{epoch:03d}_coded_stream.npy"), cod_rec)
    return cod_rec


def decoder_yuv_som(som, cod_rec, som_size):
    rows, cols = cod_rec.shape
    rec = np.zeros((rows, cols, 3), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            code = cod_rec[r, c]
            i = code // som_size
            j = code % som_size
            rec[r, c] = som.get_unit_weights(i, j)

    return rec


def reconstruct_rgb_from_indexmatrix(som, cod_rec, som_size):
    rec_som_norm = decoder_yuv_som(som, cod_rec, som_size)

    rec_yuv_q = som_output_to_yuv_q(
        rec_som_norm,
        y_bits=6,
        u_bits=4,
        v_bits=4
    )

    rec_yuv = dequantize_yuv(
        rec_yuv_q,
        y_bits=6,
        u_bits=4,
        v_bits=4
    )

    rec_rgb = yuv_to_rgb(rec_yuv)
    return rec_rgb


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
                "yuv_644_bytes_theoretical",
                "som_yuv_bytes_theoretical",
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
        next(f)
        for line in f:
            parts = line.strip().split(",")

            epoch = int(parts[0])
            mse = float(parts[5])
            psnr = float(parts[6])

            epochs.append(epoch)
            mse_values.append(mse)
            psnr_values.append(psnr)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, mse_values, marker="o")
    plt.title("MSE über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "mse_over_epochs.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnr_values, marker="o")
    plt.title("PSNR über die Epochen")
    plt.xlabel("Epoche")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "psnr_over_epochs.png"), dpi=150)
    plt.close()


def main():
    som_size = 4
    batch_size = 256
    tot_epoch = 5
    image_name = "20180110_155149.bmp"

    input_path = "./input/"
    output_path = "./output_yuv/"
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, "metrics_yuv.csv")
    save_metrics_csv_header(csv_path)

    img_original = Image.open(os.path.join(input_path, image_name)).convert("RGB")
    img_original = img_original.resize((512, 512), Image.NEAREST)

    img_uint8 = np.asarray(img_original, dtype=np.uint8)

    rows, cols, _ = img_uint8.shape
    num_colors = som_size * som_size

    # RGB → YUV
    yuv = rgb_to_yuv(img_uint8)

    # YUV quantisieren: Y = 6 Bit, U = 4 Bit, V = 4 Bit
    yuv_q = quantize_yuv(yuv, y_bits=6, u_bits=4, v_bits=4)

    # Für SOM normalisieren auf 0–1
    som_input = yuv_q_to_som_input(yuv_q, y_bits=6, u_bits=4, v_bits=4)

    som = Som(
        matrix_size=som_size,
        input_size=3,
        low=0,
        high=1,
        round_values=False
    )

    lr_decay = ExponentialDecay(0.5, decay_step=1, decay_rate=0.8, staircase=True)
    rad_decay = ExponentialDecay(2.0, decay_step=1, decay_rate=0.5, staircase=True)

    original_storage = original_rgb_storage_bits(rows, cols)
    yuv_644_storage = yuv_644_storage_bits(rows, cols)
    som_yuv_storage = yuv_som_storage_bits(rows, cols, num_colors, y_bits=6, u_bits=4, v_bits=4)

    theoretical_ratio = compression_ratio(
        original_storage["total_bytes"],
        som_yuv_storage["total_bytes"]
    )

    print("Theoretischer Speicherbedarf:")
    print(f"Original RGB:     {original_storage['total_bytes']:.2f} Bytes")
    print(f"YUV 6/4/4:        {yuv_644_storage['total_bytes']:.2f} Bytes")
    print(f"SOM-YUV codiert:  {som_yuv_storage['total_bytes']:.2f} Bytes")
    print(f"Kompressionsrate (theoretisch): {theoretical_ratio:.4f}")

    for epoch in range(1, tot_epoch + 1):
        epoch_start = time.perf_counter()

        lr = lr_decay.return_decayed_value(epoch)
        rad = rad_decay.return_decayed_value(epoch)

        batch = []
        for _ in range(batch_size):
            rr = np.random.randint(rows)
            cc = np.random.randint(cols)
            batch.append(som_input[rr, cc])

        training_start = time.perf_counter()
        som.training_batch_step(
            batch,
            learning_rate=lr,
            radius=rad,
            weighted_distance=True
        )
        training_end = time.perf_counter()
        training_time = training_end - training_start

        recon_path = os.path.join(output_path, f"epoch_{epoch:03d}_decoded_from_stream.png")

        # Indexmatrix wird EINMAL erzeugt
        cod_rec = coder(som, som_input, som_size, output_path, epoch)

        # Rekonstruktion aus derselben Indexmatrix
        rec_decoded = reconstruct_rgb_from_indexmatrix(som, cod_rec, som_size)
        Image.fromarray(rec_decoded).save(recon_path)

        save_image_and_hist(rec_decoded, epoch, output_path)
        save_som_weights_yuv_as_rgb(som, epoch, output_path)

        # RLE auf derselben Indexmatrix
        rle_values, rle_counts, rle_rows, rle_cols = rle_encode_2d(cod_rec)

        rle_path = os.path.join(output_path, f"epoch_{epoch:03d}_coded_stream_rle.npz")
        save_rle(rle_path, rle_values, rle_counts, rle_rows, rle_cols)

        cod_rec_rle_decoded = rle_decode_2d(rle_values, rle_counts, rle_rows, rle_cols)

        if not np.array_equal(cod_rec, cod_rec_rle_decoded):
            raise ValueError(f"RLE-Dekodierung fehlerhaft in Epoche {epoch}")

        rec_from_rle = reconstruct_rgb_from_indexmatrix(som, cod_rec_rle_decoded, som_size)
        Image.fromarray(rec_from_rle).save(
            os.path.join(output_path, f"epoch_{epoch:03d}_decoded_from_rle_stream.png")
        )

        rle_theoretical_bytes = estimate_rle_bytes(rle_values, rle_counts)
        rle_file_bytes = os.path.getsize(rle_path)

        mse, psnr = calculate_mse_and_psnr(img_uint8, rec_decoded)

        coded_stream_path = os.path.join(output_path, f"epoch_{epoch:03d}_coded_stream.npy")
        coded_stream_npy_bytes = os.path.getsize(coded_stream_path)
        reconstructed_png_bytes = os.path.getsize(recon_path)

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        append_metrics_row(csv_path, [
            epoch,
            round(lr, 6),
            round(rad, 6),
            round(training_time, 6),
            round(epoch_time, 6),
            round(mse, 6),
            round(psnr, 6),
            round(original_storage["total_bytes"], 2),
            round(yuv_644_storage["total_bytes"], 2),
            round(som_yuv_storage["total_bytes"], 2),
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

    som.save(output_path, "som_coder_yuv.npz")
    save_metric_plot(csv_path, output_path)

    print(f"\nFertig. Ergebnisse in: {output_path}")


if __name__ == "__main__":
    main()