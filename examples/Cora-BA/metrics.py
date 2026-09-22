import numpy as np
import math


def calculate_mse(original_array, reconstructed_array):
    original_array = np.asarray(original_array, dtype=np.float64)
    reconstructed_array = np.asarray(reconstructed_array, dtype=np.float64)

    if original_array.shape != reconstructed_array.shape:
        raise ValueError(
            f"Unterschiedliche Shapes: Original={original_array.shape}, "
            f"Rekonstruktion={reconstructed_array.shape}"
        )

    mse = np.mean((original_array - reconstructed_array) ** 2)
    return mse


def calculate_psnr_from_mse(mse, max_pixel=255.0):
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_pixel ** 2) / mse)


def calculate_snr_from_mse(mse):
    """
    Signal-Rausch-Verhältnis nach Vorgabe von Prof. Vuong

    SNR = 10 * log10((256 * 256) / MSE)
    """

    if mse == 0:
        return float("inf")

    return 10 * math.log10((256 * 256) / mse)


def calculate_mse_and_psnr(original_array, reconstructed_array):
    mse = calculate_mse(original_array, reconstructed_array)
    psnr = calculate_psnr_from_mse(mse)
    return mse, psnr


def theoretical_storage_bits(rows, cols, num_colors, include_palette=True):
    """
    Theoretischer Speicherbedarf:
    - Indexbild: ceil(log2(num_colors)) Bit pro Pixel
    - Palette: num_colors * 3 * 8 Bit
    """
    bits_per_index = int(np.ceil(np.log2(num_colors)))
    index_bits = rows * cols * bits_per_index
    palette_bits = num_colors * 3 * 8 if include_palette else 0
    total_bits = index_bits + palette_bits

    return {
        "bits_per_index": bits_per_index,
        "index_bits": index_bits,
        "palette_bits": palette_bits,
        "total_bits": total_bits,
        "total_bytes": total_bits / 8.0,
    }


def original_rgb_storage_bits(rows, cols):
    total_bits = rows * cols * 3 * 8
    return {
        "total_bits": total_bits,
        "total_bytes": total_bits / 8.0,
    }


def compression_ratio(original_bytes, compressed_bytes):
    if compressed_bytes == 0:
        return float("inf")
    return original_bytes / compressed_bytes
