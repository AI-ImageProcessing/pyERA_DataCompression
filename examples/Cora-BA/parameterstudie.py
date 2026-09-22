import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pyERA.som import Som
from pyERA.utils import ExponentialDecay

from metrics import (
    calculate_mse_and_psnr,
    calculate_snr_from_mse,
    compression_ratio,
)

from color_transform import (
    rgb_to_yuv,
    quantize_yuv,
    dequantize_yuv,
    yuv_to_rgb,
    yuv_q_to_som_input,
    som_output_to_yuv_q,
)



# Funktion: theoretischer Speicherbedarf


def som_storage_bits(rows, cols, num_colors, y_bits=6, u_bits=4, v_bits=4):
    # Bits pro Index
    bits_per_index = int(np.ceil(np.log2(num_colors)))

    # Speicher der Indexmatrix
    index_bits = rows * cols * bits_per_index

    # Speicher der LUT / SOM-Palette
    palette_bits_per_color = y_bits + u_bits + v_bits
    palette_bits = num_colors * palette_bits_per_color

    total_bits = index_bits + palette_bits

    return {
        "bits_per_index": bits_per_index,
        "total_bits": total_bits,
        "total_bytes": total_bits / 8.0,
    }



# Funktion: Decoder
# Rekonstruktion aus der Indexmatrix


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



# Funktion: Rekonstruktion RGB


def reconstruct_rgb_from_indexmatrix(som, cod_rec, som_size):

    # SOM-Ausgabe (normalisierte YUV-Werte)
    rec_som_norm = decoder_yuv_som(som, cod_rec, som_size)

    # zurück in quantisierte YUV-Werte
    rec_yuv_q = som_output_to_yuv_q(
        rec_som_norm,
        y_bits=6,
        u_bits=4,
        v_bits=4,
    )

    # Dequantisierung
    rec_yuv = dequantize_yuv(
        rec_yuv_q,
        y_bits=6,
        u_bits=4,
        v_bits=4,
    )

    # zurück nach RGB
    rec_rgb = yuv_to_rgb(rec_yuv)

    return rec_rgb



# Funktion: Indexmatrix erzeugen


def coder(som, som_input, som_size):

    rows, cols, _ = som_input.shape

    cod_rec = np.zeros((rows, cols), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            pixel = som_input[r, c]

            bmu = som.return_BMU_index(pixel)

            code = bmu[0] * som_size + bmu[1]

            cod_rec[r, c] = code

    return cod_rec



# Hauptprogramm


def main():


    # Bilder


    image_names = [
        "20180110_155149.bmp",
        "medpix2.png",
        "marilyn_filtered.jpg",
    ]


    # SOM-Größen


    som_sizes = [2, 4, 8, 16]


    # feste Parameter


    epochs = 10
    batch_size = 256

    y_bits = 6
    u_bits = 4
    v_bits = 4

    input_path = "./input/"
    output_path = "./parameterstudie_output/"

    os.makedirs(output_path, exist_ok=True)


    # CSV-Datei


    csv_path = os.path.join(output_path, "parameterstudie.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "image",
            "som_size",
            "num_colors",
            "bits_per_index",
            "theoretical_bytes",
            "compression_ratio",
            "mse",
            "psnr",
            "snr",
        ])


    # Ergebnisse pro Bild speichern


    all_results = {}


    # Schleife über alle Bilder


    for image_name in image_names:

        print(f"\nBearbeite Bild: {image_name}")

        # Bild laden
        img_original = Image.open(
            os.path.join(input_path, image_name)
        ).convert("RGB")

        img_original = img_original.resize((512, 512), Image.NEAREST)

        img_uint8 = np.asarray(img_original, dtype=np.uint8)

        rows, cols, _ = img_uint8.shape

        # RGB → YUV
        yuv = rgb_to_yuv(img_uint8)

        # Quantisierung
        yuv_q = quantize_yuv(
            yuv,
            y_bits=y_bits,
            u_bits=u_bits,
            v_bits=v_bits,
        )

        # Für SOM normalisieren
        som_input = yuv_q_to_som_input(
            yuv_q,
            y_bits=y_bits,
            u_bits=u_bits,
            v_bits=v_bits,
        )

        # Ergebnisse für die Kurve
        bits_list = []
        psnr_list = []
        snr_list = []


        # Schleife über SOM-Größen


        for som_size in som_sizes:

            print(f"  SOM-Größe: {som_size}x{som_size}")

            num_colors = som_size * som_size

            # SOM initialisieren
            som = Som(
                matrix_size=som_size,
                input_size=3,
                low=0,
                high=1,
                round_values=False,
            )

            lr_decay = ExponentialDecay(
                0.5,
                decay_step=1,
                decay_rate=0.8,
                staircase=True,
            )

            rad_decay = ExponentialDecay(
                2.0,
                decay_step=1,
                decay_rate=0.5,
                staircase=True,
            )


            # Training


            for epoch in range(1, epochs + 1):

                lr = lr_decay.return_decayed_value(epoch)
                rad = rad_decay.return_decayed_value(epoch)

                batch = []

                for _ in range(batch_size):
                    rr = np.random.randint(rows)
                    cc = np.random.randint(cols)

                    batch.append(som_input[rr, cc])

                som.training_batch_step(
                    batch,
                    learning_rate=lr,
                    radius=rad,
                    weighted_distance=True,
                )


            # EINMAL Indexmatrix erzeugen

            cod_rec = coder(som, som_input, som_size)

            # Rekonstruktion
            rec_rgb = reconstruct_rgb_from_indexmatrix(
                som,
                cod_rec,
                som_size,
            )

            # Fehler berechnen
            mse, psnr = calculate_mse_and_psnr(
                img_uint8,
                rec_rgb,
            )
            snr = calculate_snr_from_mse(mse)

            # theoretischer Speicherbedarf
            storage = som_storage_bits(
                rows,
                cols,
                num_colors,
                y_bits=y_bits,
                u_bits=u_bits,
                v_bits=v_bits,
            )

            # Original RGB
            original_bytes = rows * cols * 3

            ratio = compression_ratio(
                original_bytes,
                storage["total_bytes"],
            )

            # Werte für Plot speichern
            bits_list.append(storage["bits_per_index"])
            psnr_list.append(psnr)
            snr_list.append(snr)

            # CSV schreiben
            with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow([
                    image_name,
                    som_size,
                    num_colors,
                    storage["bits_per_index"],
                    round(storage["total_bytes"], 2),
                    round(ratio, 4),
                    round(mse, 4),
                    round(psnr, 4),
                    round(snr, 4),
                ])

            print(
                f"MSE={mse:.4f} | "
                f"PSNR={psnr:.4f} dB | "
                f"SNR={snr:.4f} dB | "
                f"Bits={storage['bits_per_index']}"
            )

        # Ergebnisse des Bildes speichern
        all_results[image_name] = {
            "bits": bits_list,
            "psnr": psnr_list,
            "snr": snr_list,
        }


    # Plot erzeugen

    plt.figure(figsize=(8, 6))

    for image_name, data in all_results.items():

        plt.plot(
            data["bits"],
            data["psnr"],
            marker="o",
            label=image_name,
        )

    plt.title("PSNR gegen Bitzahl")
    plt.xlabel("Bits pro Index")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_path, "psnr_vs_bits.png"),
        dpi=150,
    )

    plt.close()

    plt.figure(figsize=(8, 6))

    for image_name, data in all_results.items():
        plt.plot(
            data["bits"],
            data["snr"],
            marker="o",
            label=image_name,
        )

    plt.title("SNR gegen Bitzahl")
    plt.xlabel("Bits pro Index")
    plt.ylabel("SNR (dB)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_path, "snr_vs_bits.png"),
        dpi=150,
    )

    plt.close()

    print("\nParameterstudie abgeschlossen.")


if __name__ == "__main__":
    main()

