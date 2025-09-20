import collections
import numpy as np
import struct

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

# Camera model IDs and their properties
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODEL_IDS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8} etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def qvec2rotmat(qvec):
    """将四元数转换为旋转矩阵"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteImagesBinary(const std::string& path)
        void Reconstruction::ReadImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 terminator
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(
                fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D
            )
            xys = np.column_stack(
                [np.array(x_y_id_s[0::3]), np.array(x_y_id_s[1::3])]
            )
            point3D_ids = np.array(x_y_id_s[2::3])
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images
