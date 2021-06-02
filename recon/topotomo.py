import numpy as np
import pylab as plt
from matplotlib.colors import hsv_to_rgb

from recon import astra_utils


def generate_pantom():
    size = 32
    intensity = np.zeros((size, size), dtype=np.float32)
    phase = np.zeros_like(intensity)
    intensity[size // 3, size // 3:2 * size // 3] = 1.
    phase[size // 3, size // 3:2 * size // 3] = 0

    intensity[size // 3 + 1:2 * size // 3, size // 2 + 5] = 0.5
    phase[size // 3 + 1:2 * size // 3, size // 2 +5] = np.pi/2
    return intensity, phase


def show_complex_image(intensity, phase):
    H = np.squeeze(intensity)
    V = np.squeeze(phase)

    S = np.ones_like(V)
    v = V / (2 * np.pi) + 0.5
    h = (H - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    HSV = np.dstack((v, S, h))
    RGB = hsv_to_rgb(HSV)

    plt.figure(figsize=(10, 10))
    plt.imshow(RGB, interpolation='nearest')
    plt.show()


def generate_projections(intensity, phase, angles, show=False):
    if phase is not None:
        sinogram = np.zeros((angles.shape[0], intensity.shape[0]), dtype=np.float32)
        for ia, angle in enumerate(angles):
            absorbtion = intensity * np.abs(np.cos(angle + phase))
            sinogram[ia] = astra_utils.astra_fp_2d_parallel(
                absorbtion, [np.rad2deg(angle), ])
    else:
        sinogram = astra_utils.astra_fp_2d_parallel(
            intensity, np.rad2deg(angles))
    if show:
        plt.figure()
        plt.imshow(sinogram)
        plt.show()
    return sinogram


def recon_tomo(sinogram, angles, show=True):
    rec = astra_utils.astra_recon_2d_parallel(
        sinogram,
        np.rad2deg(angles),
        [["CGLS_CUDA", 50]])
    if show:
        plt.figure()
        plt.imshow(rec)
        plt.show()
    return rec


def recon_topo(sinogram, angles, show=True):
    bp = np.zeros((len(angles), sinogram.shape[1], sinogram.shape[1]), dtype=np.float32)
    for ia, angle in enumerate(angles):
        bp[ia] = astra_utils.astra_bp_2d_parallel(sinogram[None, ia], [np.rad2deg(angle),])

    # if show:
    #     ax=0
    #     for i in range(0,bp.shape[ax], 10):
    #         plt.figure()
    #         plt.imshow(bp.take(i, axis=ax))
    #         plt.show()


    rec = astra_utils.astra_recon_2d_parallel(
        sinogram,
        np.rad2deg(angles),
        [["CGLS_CUDA", 50]])

    if show:
        plt.figure()
        plt.imshow(rec)
        plt.show()
    return rec


if __name__ == "__main__":
    intensity, phase = generate_pantom()
    show_complex_image(intensity, phase)
    angles = np.linspace(0, 2*np.pi, 100)

    # sinogram_tomo = generate_projections(intensity, None, angles)
    # rec_tomo = recon_tomo(sinogram_tomo, angles)

    sinogram_topo = generate_projections(intensity, phase, angles)
    rec_topo = recon_topo(sinogram_topo, angles)

