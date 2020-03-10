import astra
import numpy as np


def build_proj_geometry_parallel_2d(detector_size, angles):
    """

    :param detector_size:
    :param angles: degrees
    :return:
    """
    detector_spacing_x = 1.0
    angles_rad = np.asarray(angles) * np.pi / 180
    proj_geom = astra.create_proj_geom('parallel',
                                       detector_spacing_x, detector_size, angles_rad)
    return proj_geom


def build_proj_geometry_fan_2d(detector_size, angles, source_object, object_det):
    """

    :param detector_size:
    :param angles: degrees
    :param source_object:
    :param object_det:
    :return:
    """
    detector_spacing_x = 1.0

    angles_rad = np.asarray(angles) * np.pi / 180
    proj_geom = astra.create_proj_geom("fanflat", detector_spacing_x, detector_size, angles_rad,
                                       source_object, object_det)
    return proj_geom


def build_volume_geometry_2d(rec_size):
    """

    :param rec_size:
    :return:
    """
    vol_geom = astra.create_vol_geom(rec_size, rec_size)
    return vol_geom


def astra_recon_2d(sinogram, proj_geom, method='FBP_CUDA', n_iters=10, data=None):
    """

    :param proj_geom:
    :param sinogram:
    :param method:
    :param n_iters:
    :param data:
    :return:
    """
    detector_size = sinogram.shape[-1]

    rec_size = detector_size
    vol_geom = build_volume_geometry_2d(rec_size)

    sinogram_id = astra.data2d.create('-sino', proj_geom, data=sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom, data)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, n_iters)
    tomo_rec = astra.data2d.get(rec_id)
    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.clear()
    return tomo_rec


def astra_bp_2d_parallel(sinogram, angles, data=None):
    detector_size = sinogram.shape[-1]
    proj_geom = build_proj_geometry_parallel_2d(detector_size, angles)
    rec = astra_recon_2d(sinogram, proj_geom, "BP_CUDA", 1, data)
    return rec


def astra_recon_2d_parallel(sinogram, angles, method='FBP_CUDA', n_iters=10, data=None):
    detector_size = sinogram.shape[-1]
    proj_geom = build_proj_geometry_parallel_2d(detector_size, angles)
    rec = astra_recon_2d(sinogram, proj_geom, method, n_iters, data)
    return rec


def astra_recon_2d_fan(sinogram, angles, source_object, object_det,
                       method='FBP_CUDA', n_iters=10, data=None):
    detector_size = sinogram.shape[-1]
    proj_geom = build_proj_geometry_fan_2d(detector_size, angles, source_object, object_det)
    rec = astra_recon_2d(sinogram, proj_geom, method, n_iters, data)
    return rec


def astra_fp_2d(volume, proj_geom):
    """
    :param proj_geom:
    :param volume:
    :return:3D sinogram
    """
    detector_size = volume.shape[1]
    rec_size = detector_size

    vol_geom = build_volume_geometry_2d(rec_size)

    sinogram_id = astra.data2d.create('-sino', proj_geom)
    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom, data=volume)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('FP_CUDA')
    cfg['VolumeDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    res_sino = astra.data2d.get(sinogram_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.clear()
    return res_sino


def astra_fp_2d_parallel(volume, angles):
    """

    :param volume:
    :param angles: degrees
    :return:
    """
    detector_size = volume.shape[1]
    proj_geom = build_proj_geometry_parallel_2d(detector_size, angles)
    rec = astra_fp_2d(volume, proj_geom)
    return rec


def astra_fp_2d_fan(volume, angles, source_object, object_det):
    """

    :param volume:
    :param angles: degrees
    :return:
    """
    detector_size = volume.shape[1]
    proj_geom = build_proj_geometry_fan_2d(detector_size, angles, source_object, object_det)
    rec = astra_fp_2d(volume, proj_geom)
    return rec


def build_volume_geometry_3d(rec_size, slices_number):
    """

    :param rec_size:
    :param slices_number:
    :return:
    """
    vol_geom = astra.create_vol_geom(rec_size, rec_size, slices_number)
    return vol_geom


def build_proj_geometry_parallel_3d(slices_number, detector_size, angles):
    """

    :param slices_number:
    :param detector_size:
    :param angles: degrees
    :return:
    """
    detector_spacing_x = 1.0
    detector_spacing_y = 1.0
    angles_rad = np.asarray(angles) * np.pi / 180
    proj_geom = astra.create_proj_geom('parallel3d',
                                       detector_spacing_x, detector_spacing_y,
                                       slices_number, detector_size, angles_rad)
    return proj_geom


def build_proj_geometry_cone_3d(slices_number, detector_size, angles, source_object, object_det):
    """

    :param slices_number:
    :param detector_size:
    :param angles: degrees
    :param source_object:
    :param object_det:
    :return:
    """
    detector_spacing_x = 1.0
    detector_spacing_y = 1.0
    angles_rad = np.asarray(angles) * np.pi / 180
    proj_geom = astra.create_proj_geom('cone', detector_spacing_x, detector_spacing_y,
                                       slices_number, detector_size, angles_rad,
                                       source_object, object_det)
    return proj_geom


def astra_fp_3d(volume, proj_geom):
    """
    :param proj_geom:
    :param volume:
    :return:3D sinogram
    """
    detector_size = volume.shape[1]
    slices_number = volume.shape[0]
    rec_size = detector_size

    vol_geom = build_volume_geometry_3d(rec_size, slices_number)

    sinogram_id = astra.data3d.create('-sino', proj_geom)
    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom, data=volume)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 1)
    res_sino = astra.data3d.get(sinogram_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinogram_id)
    astra.clear()
    return res_sino


def astra_fp_3d_parallel(volume, angles):
    """

    :param volume:
    :param angles: degrees
    :return:
    """
    detector_size = volume.shape[1]
    slices_number = volume.shape[0]
    proj_geom = build_proj_geometry_parallel_3d(slices_number, detector_size, angles)
    rec = astra_fp_3d(volume, proj_geom)
    return rec


def astra_fp_3d_cone(volume, angles, source_object, object_det):
    """

    :param volume:
    :param angles: radians
    :param source_object
    :param object_det
    :return:
    """
    detector_size = volume.shape[1]
    slices_number = volume.shape[0]
    proj_geom = build_proj_geometry_cone_3d(slices_number, detector_size, angles, source_object, object_det)
    rec = astra_fp_3d(volume, proj_geom)
    return rec


def astra_fp_3d_fan(volume, angles, source_object, object_det):
    """

    :param volume:
    :param angles: radians
    :param source_object
    :param object_det
    :return:
    """
    detector_size = volume.shape[1]
    slices_number = volume.shape[0]
    angles_number = len(angles)
    rec = np.zeros((slices_number, angles_number, detector_size), dtype='float32')
    proj_geom = build_proj_geometry_fan_2d(detector_size, angles, source_object, object_det)
    for s in range(slices_number):
        sino_t = astra_fp_2d(np.flipud(volume[s]), proj_geom)  # TODO: check why we should flipud
        rec[s] = sino_t
    return rec


def astra_recon_3d(sinogram, proj_geom, method='CGLS3D_CUDA', n_iters=10, data=None):
    """

    :param proj_geom:
    :param sinogram:
    :param method:
    :param n_iters:
    :param data:
    :return:
    """
    detector_size = sinogram.shape[-1]
    slices_number = sinogram.shape[0]

    rec_size = detector_size
    vol_geom = astra.create_vol_geom(rec_size, rec_size, slices_number)

    sinogram_id = astra.data3d.create('-sino', proj_geom, data=sinogram)

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom, data)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(method)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    cfg['option'] = {}

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, n_iters)
    tomo_rec = astra.data3d.get(rec_id)
    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(sinogram_id)
    astra.clear()
    return tomo_rec


def astra_recon_3d_parallel(sinogram, angles, method='CGLS3D_CUDA', n_iters=10, data=None):
    detector_size = sinogram.shape[2]
    slices_number = sinogram.shape[0]
    proj_geom = build_proj_geometry_parallel_3d(slices_number, detector_size, angles)
    rec = astra_recon_3d(sinogram, proj_geom, method, n_iters, data)
    return rec
