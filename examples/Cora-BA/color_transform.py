import numpy as np


def rgb_to_yuv(rgb_array):
    rgb = rgb_array.astype(np.float32)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.493 * (b - y)
    v = 0.877 * (r - y)

    return np.stack((y, u, v), axis=2)


def quantize_yuv(yuv_array, y_bits=6, u_bits=4, v_bits=4):
    y = yuv_array[:, :, 0]
    u = yuv_array[:, :, 1]
    v = yuv_array[:, :, 2]

    y_q = np.round(y / 255 * ((2 ** y_bits) - 1))

    u_q = np.round((u + 125) / 250 * ((2 ** u_bits) - 1))
    v_q = np.round((v + 157) / 314 * ((2 ** v_bits) - 1))

    y_q = np.clip(y_q, 0, (2 ** y_bits) - 1).astype(np.uint8)
    u_q = np.clip(u_q, 0, (2 ** u_bits) - 1).astype(np.uint8)
    v_q = np.clip(v_q, 0, (2 ** v_bits) - 1).astype(np.uint8)

    return np.stack((y_q, u_q, v_q), axis=2)


def dequantize_yuv(yuv_q, y_bits=6, u_bits=4, v_bits=4):
    y_q = yuv_q[:, :, 0].astype(np.float32)
    u_q = yuv_q[:, :, 1].astype(np.float32)
    v_q = yuv_q[:, :, 2].astype(np.float32)

    y = y_q / ((2 ** y_bits) - 1) * 255
    u = u_q / ((2 ** u_bits) - 1) * 250 - 125
    v = v_q / ((2 ** v_bits) - 1) * 314 - 157

    return np.stack((y, u, v), axis=2)


def yuv_to_rgb(yuv_array):
    y = yuv_array[:, :, 0]
    u = yuv_array[:, :, 1]
    v = yuv_array[:, :, 2]

    r = y + v / 0.877
    b = y + u / 0.493
    g = (y - 0.299 * r - 0.114 * b) / 0.587

    rgb = np.stack((r, g, b), axis=2)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def yuv_q_to_som_input(yuv_q, y_bits=6, u_bits=4, v_bits=4):
    y = yuv_q[:, :, 0].astype(np.float32) / ((2 ** y_bits) - 1)
    u = yuv_q[:, :, 1].astype(np.float32) / ((2 ** u_bits) - 1)
    v = yuv_q[:, :, 2].astype(np.float32) / ((2 ** v_bits) - 1)

    return np.stack((y, u, v), axis=2)


def som_output_to_yuv_q(som_output, y_bits=6, u_bits=4, v_bits=4):
    y = np.round(som_output[:, :, 0] * ((2 ** y_bits) - 1))
    u = np.round(som_output[:, :, 1] * ((2 ** u_bits) - 1))
    v = np.round(som_output[:, :, 2] * ((2 ** v_bits) - 1))

    y = np.clip(y, 0, (2 ** y_bits) - 1).astype(np.uint8)
    u = np.clip(u, 0, (2 ** u_bits) - 1).astype(np.uint8)
    v = np.clip(v, 0, (2 ** v_bits) - 1).astype(np.uint8)

    return np.stack((y, u, v), axis=2)