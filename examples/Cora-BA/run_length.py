import numpy as np


def rle_encode_1d(array_1d):
    """
    Run-Length-Encoding für ein 1D-Array.
    Rückgabe:
        values: Werte der Runs
        counts: Länge der Runs
    """
    arr = np.asarray(array_1d, dtype=np.uint8).flatten()

    if arr.size == 0:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint32)

    values = []
    counts = []

    current_value = arr[0]
    current_count = 1

    for x in arr[1:]:
        if x == current_value:
            current_count += 1
        else:
            values.append(current_value)
            counts.append(current_count)
            current_value = x
            current_count = 1

    values.append(current_value)
    counts.append(current_count)

    return np.array(values, dtype=np.uint8), np.array(counts, dtype=np.uint32)


def rle_decode_1d(values, counts):
    """
    Dekodiert die RLE-Darstellung zurück in ein 1D-Array.
    """
    values = np.asarray(values, dtype=np.uint8)
    counts = np.asarray(counts, dtype=np.uint32)

    decoded = np.repeat(values, counts)
    return decoded.astype(np.uint8)


def rle_encode_2d(index_matrix):
    """
    Kodiert eine 2D-Indexmatrix zeilenweise.
    """
    rows, cols = index_matrix.shape
    flat = index_matrix.flatten()
    values, counts = rle_encode_1d(flat)
    return values, counts, rows, cols


def rle_decode_2d(values, counts, rows, cols):
    """
    Dekodiert zurück zur 2D-Indexmatrix.
    """
    flat = rle_decode_1d(values, counts)

    if flat.size != rows * cols:
        raise ValueError(
            f"Falsche Länge nach Dekodierung: {flat.size} statt {rows * cols}"
        )

    return flat.reshape((rows, cols))


def save_rle(filepath, values, counts, rows, cols):
    np.savez_compressed(
        filepath,
        values=values,
        counts=counts,
        rows=np.array([rows], dtype=np.uint32),
        cols=np.array([cols], dtype=np.uint32)
    )


def load_rle(filepath):
    data = np.load(filepath)
    values = data["values"]
    counts = data["counts"]
    rows = int(data["rows"][0])
    cols = int(data["cols"][0])
    return values, counts, rows, cols


def estimate_rle_bytes(values, counts):
    """
    Theoretische Nutzdaten-Größe der RLE-Darstellung.
    values: uint8 -> 1 Byte pro Eintrag
    counts: uint32 -> 4 Byte pro Eintrag
    """
    return values.nbytes + counts.nbytes