import contextlib
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Tuple
import lovely_logger as logging
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests

import streamlit as st
from pandas import read_csv
from scipy import stats
from shapely.geometry import LineString, Point, Polygon


# shapely.geometry.polygon.orient
from sklearn.neighbors import KDTree

# name
# trajectory
# geometry

examples = {
    # , free choice of destination
    "Bidirectional corridor (exp)": [
        "Bi-direct",
        "https://fz-juelich.sciebo.de/s/o4D8Va2MtbSeG2v/download",
        "https://fz-juelich.sciebo.de/s/FNuSYwOre85km3U/download",
    ],
    "Bottleneck BUW (exp)": [
        "030_c_56_h0",
        "https://fz-juelich.sciebo.de/s/AsrA465S3wNDNlo/download",
        "https://fz-juelich.sciebo.de/s/rVdksQ7yUngiUmw/download",
    ],
    "Bottleneck WDG (exp)": [
        "WDG_09",
        "https://fz-juelich.sciebo.de/s/oTG7vRCcQyYJ08q/download",
        "https://fz-juelich.sciebo.de/s/lDuCQlJkwh9Of1C/download",
    ],
    "Corner (exp)": [
        "jps_eo-300-300-300_combined_MB",
        "https://fz-juelich.sciebo.de/s/BfNxMk1qM64QqYj/download",
        "https://fz-juelich.sciebo.de/s/qNVoD8RZ8UentBB/download",
    ],
    "Crossing 90 (exp)": [
        "CROSSING_90_a_10",
        "https://fz-juelich.sciebo.de/s/gLfaofmZCNtf5Vx/download",
        "https://fz-juelich.sciebo.de/s/f960CoXb26FKpkw/download",
    ],
    "Crossing 120 (exp)": [
        "CROSSING_120_A_1",
        "https://fz-juelich.sciebo.de/s/X3WTuExdj2HXRVx/download",
        "https://fz-juelich.sciebo.de/s/11Cz0bQWZCv23eI/download",
    ],
    "Crossing 120  (exp)": [
        "CROSSING_120_C_1",
        "https://fz-juelich.sciebo.de/s/vrkGlCDKVTIz8Ch/download",
        "https://fz-juelich.sciebo.de/s/11Cz0bQWZCv23eI/download",
    ],
    "Stadium Entrance (exp)": [
        "mo11_combine_MB",
        "https://fz-juelich.sciebo.de/s/ckzZLnRJCKKgAnZ/download",
        "https://fz-juelich.sciebo.de/s/kgXUEyu95FTQlFC/download",
    ],
    "Multi-Rooms (sim)": [
        "multi-rooms",
        "https://fz-juelich.sciebo.de/s/7kwrnAzcv5m7ii2/download",
        "https://fz-juelich.sciebo.de/s/VSPgE6Kcfp8qDIa/download",
    ],
    "Bottleneck (sim)": [
        "bottleneck",
        "https://fz-juelich.sciebo.de/s/HldXLySEfEDMdZo/download",
        "https://fz-juelich.sciebo.de/s/FqiSFGr6FajfYLD/download",
    ],
    "HC-BUW (sim)": [
        "HC_BUW",
        "https://fz-juelich.sciebo.de/s/GgvVjc81lzmhTgv/download",
        "https://fz-juelich.sciebo.de/s/NikHJ6TIHCwSoUM/download",
    ],
}


def get_time(t):
    """Time in min sec

    :param t: Run time
    :type t: float
    :returns: str

    """

    minutes = t // 60
    seconds = t % 60
    return f"""{minutes:.0f} min:{seconds:.0f} sec"""


def selected_traj_geo(text):
    """Returns a list of trajectory and geometry files"""
    if text in examples:
        return examples[text]

    return []


def download(url: str, filename: str):

    try:
        r = requests.get(url, stream=True, timeout=10)
        logging.info(f"saving to {filename}")
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

    except Exception as e:
        st.error(
            f"""Download of file {filename} failed.\n
            Error: {e}"""
        )


@contextlib.contextmanager
def profile(name):
    start_time = time.time()
    yield  # <-- your code will execute here
    total_time = time.time() - start_time
    logging.info(f"{name}: {total_time * 1000.0:.4f} ms")


def weidmann(rho, v0=1.34, rho_max=5.4, gamma=1.913):
    inv_rho = np.empty_like(rho)
    mask = rho <= 0.01
    inv_rho[mask] = 1 / rho_max
    inv_rho[~mask] = 1 / rho[~mask]
    return v0 * (1 - np.exp(-gamma * (inv_rho - 1 / rho_max)))  # Eq. 6


def inv_weidmann(v, v0=1.34, rho_max=5.4, gamma=1.913):
    v0 = np.max(v)
    v[v > v0] = v0
    s = 1 - v / v0
    # np.log(s, where=np.logical_not(zero_mask))
    x = -1 / gamma * np.log(s, out=np.zeros_like(s), where=s != 0) + 1 / rho_max
    return 1 / x


def get_speed_index(traj_file):
    lines = traj_file[:500].split("\n")
    for line in lines:
        if line.startswith("#ID"):
            if "V" in line:
                return int(line.split().index("V"))

    return -1


def get_header(traj_file):
    lines = traj_file[:500].split("\n")
    for line in lines:
        if line.startswith("#ID"):
            if "FR" in line:
                return line

    return "Not extracted"


# todo: update with more rules for more files
def get_fps(traj_file):
    fps = traj_file.split("framerate:")[-1].split("\n")[0]
    try:
        fps = int(float(fps))
    except ValueError:
        st.error(f"{fps} in header can not be converted to int")
        logging.error(f"{fps} in header can not be converted to int")
        st.stop()

    return fps


def detect_jpscore(traj_file):
    return "#description: jpscore" in traj_file


def get_index_group(traj_file):
    index = -1
    lines = traj_file.split("\n")
    for line in lines:
        if "ID" in line and "GROUP" in line:
            line_split = line.split()
            for elem in line_split:
                index += 1
                if elem == "GROUP":
                    return index

    return index


def get_unit(traj_file):
    unit = "NOTHING"
    if "#description: jpscore" in traj_file:
        unit = "m"
    else:
        # petrack
        unit_list = traj_file.split("unit:")
        if len(unit_list) > 1:
            unit = unit_list[-1].split("\n")[0]

        if "x/" in traj_file:
            unit = traj_file.split("x/")[-1].split()[0]

        if "<x>" in traj_file:
            # <number> <frame> <x> [in m] <y> [in m] <z> [in m]
            unit = traj_file.split("<x>")[-1].split()[1].strip("]")

    unit = unit.strip()
    logging.info(f"Unit detected: <{unit}>")
    return unit


def get_transitions(xml_doc, unit):
    if unit == "cm":
        cm2m = 100
    else:
        cm2m = 1

    transitions = {}
    for _, t_elem in enumerate(xml_doc.getElementsByTagName("transition")):
        Id = t_elem.getAttribute("id")
        n_vertex = len(t_elem.getElementsByTagName("vertex"))
        vertex_array = np.zeros((n_vertex, 2))
        for v_num, _ in enumerate(t_elem.getElementsByTagName("vertex")):
            vertex_array[v_num, 0] = (
                t_elem.getElementsByTagName("vertex")[v_num].attributes["px"].value
            )
            vertex_array[v_num, 1] = (
                t_elem.getElementsByTagName("vertex")[v_num].attributes["py"].value
            )

        transitions[Id] = vertex_array / cm2m

    return transitions


def get_measurement_lines(xml_doc, unit):
    """add area_L

    https://www.jupedsim.org/jpsreport_inifile#measurement-area
    """

    if unit == "cm":
        cm2m = 100
    else:
        cm2m = 1

    fake_id = 1000
    measurement_lines = {}
    for _, t_elem in enumerate(xml_doc.getElementsByTagName("area_L")):
        Id = t_elem.getAttribute("id")
        logging.info(f"Measurement id = <{Id}>")
        if Id == "":
            st.warning(f"Got Measurement line with no Id. Setting id = {fake_id}")
            logging.info(f"Got Measurement line with no Id. Setting id = {fake_id}")
            Id = fake_id
            fake_id += 1

        n_vertex = 2
        vertex_array = np.zeros((n_vertex, 2))
        vertex_array[0, 0] = (
            t_elem.getElementsByTagName("start")[0].attributes["px"].value
        )
        vertex_array[0, 1] = (
            t_elem.getElementsByTagName("start")[0].attributes["py"].value
        )
        vertex_array[1, 0] = (
            t_elem.getElementsByTagName("end")[0].attributes["px"].value
        )
        vertex_array[1, 1] = (
            t_elem.getElementsByTagName("end")[0].attributes["py"].value
        )
        measurement_lines[Id] = vertex_array / cm2m
        logging.info(f"vertex: {vertex_array}")
    return measurement_lines


# def passing_frame(
#     ped_data: np.array, line: LineString, fps: int, max_distance: float
# ) -> int:
#     """Return frame of first time ped enters the line buffer

#     Enlarge the line by eps, a constant that is dependent on fps
#     eps = 1/fps * v0, v0 = 1.3 m/s

#     :param ped_data: trajectories of ped
#     :param line: transition
#     :param fps: fps
#     : param max_distance: an arbitrary distance to the line

#     :returns: frame of entrance. Return negative number if ped did not pass trans

#     """
#     eps = 1 / fps * 1.3
#     line_buffer = line.buffer(eps, cap_style=3)
#     p = ped_data[np.abs(ped_data[:, 2] - line.centroid.x) < max_distance]
#     for (frame, x, y) in p[:, 1:4]:
#         if Point(x, y).within(line_buffer):
#             return frame

#     return -1


# def passing_frame2(ped_data, line: LineString, fps: int, max_distance: float) -> int:
#     s = STRtree([Point(ped_data[i, 2:4]) for i in range(ped_data.shape[0])])
#     index = s.nearest_item(line)
#     # nearest_point = ped_data[index, 2:4]
#     nearest_frame = ped_data[index, 1]
#     # print("nearest: ", nearest_point, "at", nearest_frame)
#     L1 = line.coords[0]
#     L2 = line.coords[1]
#     P1 = ped_data[0, 2:4]
#     P2 = ped_data[-1, 2:4]
#     # print("Ped", P1, P2)
#     # print("Line", L1, L2)
#     sign1 = np.cross([L1, L2], [L1, P1])[1]
#     sign2 = np.cross([L1, L2], [L1, P2])[1]

#     if np.sign(sign1) != np.sign(sign2):
#         # crossed_line = True
#         return nearest_frame

#     # crossed_line = False
#     return -1
#     # print("nearest_frame", nearest_frame)
#     # print("Crossed?", crossed_line)


def on_different_sides(L1, L2, P1, P2) -> bool:
    """True is P1 and P2 are on different sides from [L1, L2]

                        L1
                        x
                        |
                        |
                  P1 x  |     x P2
                        x
                        L2
    --> True
    """

    sign1 = np.cross(L1 - L2, L1 - P1)
    sign2 = np.cross(L1 - L2, L1 - P2)
    return np.sign(sign1) != np.sign(sign2)


def passing_frame(
    ped_data: npt.NDArray[np.float64], line: LineString, fps: float
) -> Tuple[int, int]:
    """First frame at which the pedestrian is within a buffer around line

    fps is used to determin the width of the buffer and is not needed
    in the calculations.
    Assume a desired speed of 1.3 m/s

    :param ped_data: trajectories
    :type ped_data: np.array
    :param line: measurement line
    :type line: LineString
    :param fps: frames per second.
    :type fps: float
    :returns:

    """
    XY = ped_data[:, 2:4]
    L1: np.ndarray[int, np.dtype[np.float64]] = np.array(line.coords[0])
    L2: np.ndarray[int, np.dtype[np.float64]] = np.array(line.coords[1])
    P1 = XY[0]
    P2 = XY[-1]
    i1 = 0  # index of first element
    i2 = len(XY) - 1  # index of last element
    im = int(len(XY) / 2)  # index of the element in the middle
    M = XY[im]
    i = 0
    passed_line_at_frame = -1
    sign = -1
    if not on_different_sides(L1, L2, P1, P2):
        return passed_line_at_frame, sign

    while i1 + 1 < i2 and i < 20:
        i += 1  # to avoid endless loops! Should be removed!
        if on_different_sides(L1, L2, M, P2):
            P1 = M
            i1 = im
        else:
            P2 = M
            i2 = im

        im = int((i1 + i2) / 2)
        M = XY[im]

    # this is to ensure, that the pedestrian really passed *through* the line
    line_buffer = line.buffer(1.3 / fps, cap_style=2)
    if Point(XY[i1]).within(line_buffer):
        passed_line_at_frame = ped_data[i1, 1]
        sign = np.sign(np.cross(L1 - L2, XY[i1] - XY[i2]))
    elif Point(XY[i2]).within(line_buffer):
        passed_line_at_frame = ped_data[i2, 1]
        sign = np.sign(np.cross(L1 - L2, XY[i1] - XY[i2]))

    return passed_line_at_frame, sign


def read_trajectory(input_file):
    data = read_csv(input_file, sep=r"\s+", dtype=np.float64, comment="#").values
    return data


def read_obstacle(xml_doc, unit):
    if unit == "cm":
        cm2m = 100
    else:
        cm2m = 1

    # Initialization of a dictionary with obstacles
    return_dict = {}
    # read in obstacles and combine
    # them into an array for polygon representation
    points = np.zeros((0, 2))
    for o_num, o_elem in enumerate(xml_doc.getElementsByTagName("obstacle")):
        N_polygon = len(o_elem.getElementsByTagName("polygon"))
        if N_polygon == 1:
            pass
        else:
            points = np.zeros((0, 2))

        for _, p_elem in enumerate(o_elem.getElementsByTagName("polygon")):
            for _, v_elem in enumerate(p_elem.getElementsByTagName("vertex")):
                vertex_x = float(
                    # p_elem.getElementsByTagName("vertex")[v_num].attributes["px"].value
                    v_elem.attributes["px"].value
                )
                vertex_y = float(
                    # p_elem.getElementsByTagName("vertex")[v_num].attributes["py"].value
                    v_elem.attributes["py"].value
                )
                points = np.vstack([points, [vertex_x / cm2m, vertex_y / cm2m]])

        points = np.unique(points, axis=0)
        x = points[:, 0]
        y = points[:, 1]
        n = len(points)
        center_point = [np.sum(x) / n, np.sum(y) / n]
        angles = np.arctan2(x - center_point[0], y - center_point[1])
        # sorting the points:
        sort_tups = sorted(list(zip(x, y, angles)), key=lambda t: t[2])
        return_dict[o_num] = np.array(sort_tups)[:, 0:2]

    return return_dict


def read_subroom_walls(xml_doc, unit):
    dict_polynom_wall = {}
    n_wall = 0
    if unit == "cm":
        cm2m = 100
    else:
        cm2m = 1

    for _, s_elem in enumerate(xml_doc.getElementsByTagName("subroom")):
        for _, p_elem in enumerate(s_elem.getElementsByTagName("polygon")):
            # if p_elem.getAttribute("caption") == "wall":
            n_wall = n_wall + 1
            n_vertex = len(p_elem.getElementsByTagName("vertex"))
            vertex_array = np.zeros((n_vertex, 2))
            for v_num, _ in enumerate(p_elem.getElementsByTagName("vertex")):
                vertex_array[v_num, 0] = (
                    p_elem.getElementsByTagName("vertex")[v_num].attributes["px"].value
                )
                vertex_array[v_num, 1] = (
                    p_elem.getElementsByTagName("vertex")[v_num].attributes["py"].value
                )

            dict_polynom_wall[n_wall] = vertex_array / cm2m

    return dict_polynom_wall


def geo_limits(geo_xml, unit):
    geometry_wall = read_subroom_walls(geo_xml, unit)
    geominX = 1000
    geomaxX = -1000
    geominY = 1000
    geomaxY = -1000
    Xmin = []
    Ymin = []
    Xmax = []
    Ymax = []
    for _, wall in geometry_wall.items():
        Xmin.append(np.min(wall[:, 0]))
        Ymin.append(np.min(wall[:, 1]))
        Xmax.append(np.max(wall[:, 0]))
        Ymax.append(np.max(wall[:, 1]))

    geominX = np.min(Xmin)
    geomaxX = np.max(Xmax)
    geominY = np.min(Ymin)
    geomaxY = np.max(Ymax)
    return geominX, geomaxX, geominY, geomaxY


def get_geometry_file(traj_file):
    return traj_file.split("geometry:")[-1].split("\n")[0].strip()


def touch_default_geometry_file(data, _unit, geo_file):
    """Creates a bounding box around the trajectories

    :param data: 2D-array
    :param Unit: Unit of the trajectories (cm or m)
    :param geo_file: write geometry in this file
    :returns: geometry file named geometry.xml

    """
    # ----------
    delta = 100 if _unit == "cm" else 1
    # 1 m around to better contain the trajectories
    xmin = np.min(data[:, 2]) - delta
    xmax = np.max(data[:, 2]) + delta
    ymin = np.min(data[:, 3]) - delta
    ymax = np.max(data[:, 3]) + delta
    # --------
    # create_geo_header
    data = ET.Element("geometry")
    data.set("version", "0.8")
    data.set("caption", "experiment")
    data.set("unit", "m")  # jpsvis does not support another unit!
    # make room/subroom
    rooms = ET.SubElement(data, "rooms")
    room = ET.SubElement(rooms, "room")
    room.set("id", "0")
    room.set("caption", "room")
    subroom = ET.SubElement(room, "subroom")
    subroom.set("id", "0")
    subroom.set("caption", "subroom")
    subroom.set("class", "subroom")
    subroom.set("A_x", "0")
    subroom.set("B_y", "0")
    subroom.set("C_z", "0")
    # poly1
    polygon = ET.SubElement(subroom, "polygon")
    polygon.set("caption", "wall")
    polygon.set("type", "internal")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmin}")
    vertex.set("py", f"{ymin}")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmax}")
    vertex.set("py", f"{ymin}")
    # poly2
    polygon = ET.SubElement(subroom, "polygon")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmax}")
    vertex.set("py", f"{ymin}")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmax}")
    vertex.set("py", f"{ymax}")
    # poly3
    polygon = ET.SubElement(subroom, "polygon")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmax}")
    vertex.set("py", f"{ymax}")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmin}")
    vertex.set("py", f"{ymax}")
    # poly4
    polygon = ET.SubElement(subroom, "polygon")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmin}")
    vertex.set("py", f"{ymax}")
    vertex = ET.SubElement(polygon, "vertex")
    vertex.set("px", f"{xmin}")
    vertex.set("py", f"{ymin}")
    b_xml = ET.tostring(data, encoding="utf8", method="xml")

    with open(geo_file, "wb") as f:
        f.write(b_xml)


def compute_speed(data, fps, df=10):
    """Calculates the speed and the angle from the trajectory points.

    Using the forward formula
    speed(f) = (X(f+df) - X(f))/df [1]
    note: The last df frames are not calculated using [1].
    It is assumes that the speed in the last frames
    does not change
    :param traj: trajectory of ped (x, y). 2D array
    :param df: number of frames forwards
    :param fps: frames per seconds

    :returns: speed, angle

    example:
    df=4, S=10
         0 1 2 3 4 5 6 7 8 9
       X * * * * * * * * * *
       V + + + + + +
         *       *
           *       *      X[df:]
    X[:S-df] *       *       │
    │          *       *   ◄─┘
    └────────►   *       *
                   *       *
    """
    agents = np.unique(data[:, 0]).astype(int)
    once = 1
    speeds = np.array([])
    for agent in agents:
        ped = data[data[:, 0] == agent]
        traj = ped[:, 2:4]
        size = traj.shape[0]
        speed = np.ones(size)
        if size < df:
            logging.warning(
                f"""Compute_speed: The number of frames used to calculate the speed {df}
                exceeds the total amount of frames ({size}) in this trajectory."""
            )
            st.error(
                f"""Compute_speed: The number of frames used to calculate the speed {df}
                exceeds the total amount of frames ({size}) in this trajectory."""
            )
            st.stop()

        delta = traj[df:, :] - traj[: size - df, :]
        delta_square = np.square(delta)
        delta_x_square = delta_square[:, 0]
        delta_y_square = delta_square[:, 1]
        s = np.sqrt(delta_x_square + delta_y_square)
        speed[: size - df] = s / df * fps
        speed[size - df :] = speed[size - df - 1]
        if once:
            speeds = speed
            once = 0
        else:
            speeds = np.hstack((speeds, speed))

    return speeds


def compute_speed_and_angle(data, fps, df=10):
    """Calculates the speed and the angle from the trajectory points.

    Using the forward formula
    speed(f) = (X(f+df) - X(f))/df [1]
    note: The last df frames are not calculated using [1].
    It is assumes that the speed in the last frames
    does not change
    :param traj: trajectory of ped (x, y). 2D array
    :param df: number of frames forwards
    :param fps: frames per seconds

    :returns: speed, angle

    example:
    df=4, S=10
         0 1 2 3 4 5 6 7 8 9
       X * * * * * * * * * *
       V + + + + + +
         *       *
           *       *      X[df:]
    X[:S-df] *       *       │
    │          *       *   ◄─┘
    └────────►   *       *
                   *       *
    """
    agents = np.unique(data[:, 0]).astype(int)
    once = 1
    data2 = np.array([])
    for agent in agents:
        ped = data[data[:, 0] == agent]
        traj = ped[:, 2:4]
        size = traj.shape[0]
        speed = np.ones(size)
        angle = np.zeros(size)

        if size < df:
            logging.warning(
                f"""Compute_speed_and_angle() The number of frames used to calculate the speed {df}
                exceeds the total amount of frames ({size}) for pedestrian {agent}"""
            )
            st.error(
                f"""Compute_speed_and_angle() The number of frames used to calculate the speed {df}
                exceeds the total amount of frames ({size}) for pedestrian {agent}"""
            )
        else:
            delta = traj[df:, :] - traj[: size - df, :]
            delta_x = delta[:, 0]
            delta_y = delta[:, 1]

            delta_square = np.square(delta)
            delta_x_square = delta_square[:, 0]
            delta_y_square = delta_square[:, 1]
            angle[: size - df] = np.arctan2(delta_y, delta_x) * 180 / np.pi

            s = np.sqrt(delta_x_square + delta_y_square)
            speed[: size - df] = s / df * fps
            speed[size - df :] = speed[size - df - 1]
            angle[size - df :] = angle[size - df - 1]

        ped = np.hstack((ped, angle.reshape(size, 1)))
        ped = np.hstack((ped, speed.reshape(size, 1)))
        if once:
            data2 = ped
            once = 0
        else:
            data2 = np.vstack((data2, ped))

    return data2


def calculate_speed_average(geominX, geomaxX, geominY, geomaxY, dx, dy, X, Y, speed):
    """Calculate speed average over time"""
    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dy, dy)
    ret = stats.binned_statistic_2d(
        X,
        Y,
        speed,
        "mean",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T)


def calculate_density_average_weidmann(
    geominX, geomaxX, geominY, geomaxY, dx, dy, X, Y, speed
):
    """Calculate density using Weidmann(speed)"""
    density = inv_weidmann(speed)
    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dy, dy)
    ret = stats.binned_statistic_2d(
        X,
        Y,
        density,
        "mean",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T)  # / nframes


def calculate_density_average_classic(
    geominX, geomaxX, geominY, geomaxY, dx, dy, nframes, X, Y
):
    """Calculate classical method

    Density = mean_time(N/A_i)
    """

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dy, dy)
    area = dx * dy
    ret = stats.binned_statistic_2d(
        X,
        Y,
        None,
        "count",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T) / nframes / area


def calculate_density_frame_classic(geominX, geomaxX, geominY, geomaxY, dx, dy, X, Y):
    """Calculate classical method

    Density = mean_time(N/A_i)
    """

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dy, dy)
    area = dx * dy
    ret = stats.binned_statistic_2d(
        X,
        Y,
        None,
        "count",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T) / area


def calculate_RSET(geominX, geomaxX, geominY, geomaxY, dx, dy, X, Y, time, func):
    """Calculate RSET according to 5.5.1 RSET Maps in Schroder2017a"""
    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dy, dy)
    ret = stats.binned_statistic_2d(
        X,
        Y,
        time,
        func,
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T)


def check_shape_and_stop(shape, how_speed):
    """Write an error message if shape < 10 and stop"""
    if shape < 10 and how_speed == "from simulation":
        st.error(
            f"""Trajectory file does not have enough columns ({shape} < 10).
            \n Use <optional_output   speed=\"TRUE\">\n
            https://www.jupedsim.org/jpscore_inifile.html#header
            \n or choose option `"from trajectory"`
            """
        )
        st.stop()


def width_gaussian(fwhm):
    """np.sqrt(2) / (2 * np.sqrt(2 * np.log(2)))"""

    return fwhm * 0.6005612


def Gauss(x, a):
    """1 / (np.sqrt(np.pi) * a) * np.e ** (-x ** 2 / a ** 2)"""

    return 1 / (1.7724538 * a) * np.e ** (-(x ** 2) / a ** 2)


def densityField(x_dens, y_dens, a):
    rho_matrix_x = Gauss(x_dens, a)
    rho_matrix_y = Gauss(y_dens, a)
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.T


def xdensYdens(lattice_x, lattice_y, x_array, y_array):
    x_dens = np.add.outer(-x_array, lattice_x)
    y_dens = np.add.outer(-y_array, lattice_y)
    return x_dens, y_dens


def calculate_density_average_gauss(
    geominX, geomaxX, geominY, geomaxY, dx, dy, nframes, width, X, Y
):
    """
    Calculate density using Gauss method
    """

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dy, dy)
    x_dens, y_dens = xdensYdens(X, Y, xbins, ybins)
    a = width_gaussian(width)
    rho_matrix = densityField(x_dens, y_dens, a) / nframes
    return rho_matrix


def jam_frames(data, jam_speed):
    """Definition of jam

    return data in jam
    """

    jam_data = data[data[:, st.session_state.speed_index] <= jam_speed]
    return np.unique(jam_data[:, 1])


def consecutive_chunks(data1d, frame_margin):
    # input array([ 1,  2,  3,  4, 10, 11, 12, 15])
    # output array([3, 2])
    # diff err by 5 frames

    data1d = np.hstack(([0], data1d, [0]))
    # print("data")
    # print(data1d)
    consecutive = np.diff(data1d, 1)

    condition = consecutive == 1

    if not condition.any():
        return np.array([]), np.array([])

    if condition[0]:
        condition = np.concatenate([[False], condition])

    idx = np.where(~condition)[0]
    chunks = np.ediff1d(idx) - 1

    # print(data1d[idx])
    # --
    ret = []
    idx = idx - 1
    # print("idx", idx)
    for i, e in enumerate(idx):
        From = e + 1
        if i < len(idx) - 1:
            # print(i)
            To = idx[i + 1]
            # print(f"From = {From}, To = {To} ({To-From-np.max(chunks)}), Max = {np.max(chunks)}")
            cond = np.abs(To - From - np.max(chunks)) <= frame_margin
            if From < To and cond:
                ret.append([From, To + 1])

    # logging.info(f"Jam chunks {chunks}. From-To: {ret}, Max = {np.max(chunks)}")
    return chunks, np.array(ret)


def jam_waiting_time(
    data: npt.NDArray[np.float64],
    jam_speed: float,
    jam_min_duration: int,
    fps: int,
    precision,
) -> npt.NDArray[np.float64]:
    """Return a list of pid and its max_time in jam

    return a 2D array [ped, waiting_time]
    """
    waiting_times = []
    peds = np.unique(data[:, 0]).astype(int)
    for ped in peds:
        data_ped = data[data[:, 0] == ped]
        frames_in_jam = jam_frames(data_ped, jam_speed)
        jam_times, _ = consecutive_chunks(frames_in_jam, precision)

        if not jam_times.size:
            continue

        max_waiting_time = np.max(jam_times) / fps
        if max_waiting_time >= jam_min_duration:
            waiting_times.append([ped, max_waiting_time])

    return np.array(waiting_times)


def jam_lifetime(
    data: npt.NDArray[np.float64],
    jam_frames,
    jam_min_agents: int,
    fps: int,
    precision: int,
):
    """Lifespane of a Jam and how many pedestrian in chunck"""

    lifetime = []
    all_frames = np.unique(data[:, 1]).astype(int)
    # frame, num peds in jam. Using only the first,
    # since I dont know yet how to use the second
    # Ignore the first frames, where agents start from 0 (so in jam)
    for frame in all_frames:
        if frame in jam_frames:
            d = data[data[:, 1] == frame]
            num_ped_in_jam = len(d[:, 0])
            if num_ped_in_jam >= jam_min_agents:
                lifetime.append([frame, num_ped_in_jam])

    lifetime = np.array(lifetime)

    if not lifetime.size:
        return np.array([]), np.array([]), 0, np.array([])

    chuncks, ret = consecutive_chunks(np.array(lifetime)[:, 0], precision)
    # print("clifetime ", clifetime)
    if not chuncks.size:  # one big chunk
        chuncks = lifetime[:, 0]
        mx_lt = (np.max(chuncks) - np.min(chuncks)) / fps
    else:
        mx_lt = np.max(chuncks) / fps

    # print("F lifetime", lifetime[:10, :], lifetime.shape)
    return lifetime, chuncks, mx_lt, ret


def calculate_NT_data(transitions, selected_transitions, data, fps):
    """Frame and cumulative number of pedestrian passing transitions.

    return:
    Frame
    cum_num
    trans_used
    max_len (len of longest vector)
    Needed to stack arrays and save them in file
    """
    tstats = defaultdict(list)
    # bidirectional flow
    cum_num_negativ = {}
    cum_num_positiv = {}
    cum_num = {}
    msg = ""
    trans_used = {}

    peds = np.unique(data[:, 0]).astype(int)
    with st.spinner("Processing ..."):
        max_len = -1
        for i, t in transitions.items():
            trans_used[i] = False
            if i in selected_transitions:
                line = LineString(t)
                # len_line = line.length
                for ped in peds:
                    ped_data = data[data[:, 0] == ped]
                    # frame = passing_frame(ped_data, line, fps, len_line)
                    frame, sign = passing_frame(ped_data, line, fps)
                    if frame >= 0:
                        tstats[i].append([ped, frame, sign])
                        trans_used[i] = True

                if trans_used[i]:
                    tstats[i] = np.array(tstats[i])
                    tstats[i] = tstats[i][tstats[i][:, 1].argsort()]  # sort by frame
                    arrivals = tstats[i][:, 1]
                    arrivals_positiv = arrivals[tstats[i][:, 2] == 1]
                    arrivals_negativ = arrivals[tstats[i][:, 2] == -1]
                    cum_num[i] = np.cumsum(np.ones(len(arrivals)))
                    inx_pos = np.in1d(arrivals, arrivals_positiv)
                    inx_neg = np.in1d(arrivals, arrivals_negativ)
                    tmp_positiv = np.zeros(len(arrivals))
                    tmp_positiv[inx_pos] = 1
                    tmp_negativ = np.zeros(len(arrivals))
                    tmp_negativ[inx_neg] = 1

                    cum_num_positiv[i] = np.cumsum(tmp_positiv)
                    cum_num_negativ[i] = np.cumsum(tmp_negativ)
                    flow = (cum_num[i][-1] - 1) / (arrivals[-1] - arrivals[0]) * fps
                    if arrivals_positiv.size:
                        flow_positiv = (
                            (cum_num_positiv[i][-1] - 1)
                            / (arrivals_positiv[-1] - arrivals_positiv[0])
                            * fps
                        )
                    else:
                        flow_positiv = 0

                    if arrivals_negativ.size:
                        flow_negativ = (
                            (cum_num_negativ[i][-1] - 1)
                            / (arrivals_negativ[-1] - arrivals_negativ[0])
                            * fps
                        )
                    else:
                        flow_negativ = 0
                    # with profile("rolling flow: "):
                    #     mean_flow, std_flow = rolling_flow(arrivals, fps, windows=100)

                    max_len = max(
                        max_len, cum_num_positiv[i].size, cum_num_negativ[i].size
                    )
                    msg += f"Transition {i}: length {line.length:.2f}, flow+: {flow_positiv:.2f}, flow-: {flow_negativ:.2f} flow: {flow:.2f} [1/s],  specific flow: {flow/line.length:.2f} [1/s/m] \n \n"
                else:
                    msg += (
                        f"Transition {i}: length {line.length:.2f}, flow: 0 [1/s] \n \n"
                    )

    return tstats, cum_num, cum_num_positiv, cum_num_negativ, trans_used, max_len, msg


#  empirical CDF P(x<=X)
def CDF(x, times):
    return float(len(times[times <= x])) / len(times)


def survival(times):
    diff = np.diff(times)
    diff = np.sort(diff)
    vF = np.vectorize(CDF, excluded=["times"])
    y_diff = 1 - vF(x=diff, times=diff)
    return y_diff, diff


def rolling_flow(times, fps, windows=200):
    times = np.sort(times)
    serie = pd.Series(times)
    minp = 100
    # windows = 200 #int(len(times)/10);
    minp = min(minp, windows)
    flow = (
        fps
        * (windows - 1)
        / (
            serie.rolling(windows, min_periods=minp).max()
            - serie.rolling(windows, min_periods=minp).min()
        )
    )
    flow = flow[~np.isnan(flow)]  # remove NaN
    wmean = flow.rolling(windows, min_periods=minp).mean()
    wstd = flow.rolling(windows, min_periods=minp).std()
    return np.mean(wmean), np.mean(wstd)


def peds_inside(data):
    p_inside = []
    frames = np.unique(data[:, 1])
    for frame in frames:
        d = data[data[:, 1] == frame][:, 0]
        p_inside.append(len(d))

    return p_inside


def get_neighbors_at_frame(frame, data, k):
    at_frame = data[data[:, 1] == frame]
    points = at_frame[:, 2:4]
    tree = KDTree(points)
    if k < len(points):
        nearest_dist, nearest_ind = tree.query(points, k)
        return nearest_dist, nearest_ind

    return np.array([]), np.array([])


def get_neighbors_special_agent_data(agent, frame, data, nearest_dist, nearest_ind):

    at_frame = data[data[:, 1] == frame]
    points = at_frame[:, 2:4]
    _speeds = at_frame[:, st.session_state.speed_index]
    Ids = at_frame[:, 0]
    if (at_frame[:, 0] == agent).any():
        agent_index = np.where(at_frame[:, 0] == agent)[0][0]
        mask = nearest_ind[:, 0] == agent_index
        neighbors_ind = nearest_ind[mask][0, 1:]
        neighbors_dist = nearest_dist[mask][0, 1:]
        neighbors = np.array([points[i] for i in neighbors_ind])
        neighbors_ids = np.array([Ids[i] for i in neighbors_ind])
        neighbors_speeds = np.array([_speeds[i] for i in neighbors_ind])
    else:
        return np.array([]), np.array([]), 0, np.array([]), np.array([])

    if len(neighbors) > 2:
        my_polygon = Polygon(neighbors)
        # my_polygon2 = shapely.geometry.polygon.orient(my_polygon, sign=1.0)
        neighbors = np.array(my_polygon.exterior.coords)
        area = my_polygon.area
    else:
        area = 0

    return neighbors, neighbors_ids, area, neighbors_dist, neighbors_speeds


def get_neighbors_pdf(nearest_dist):
    distances = np.unique(nearest_dist)
    loc = distances.mean()
    scale = distances.std()
    distances = np.hstack(([0, 0], distances))
    pdf = stats.norm.pdf(distances, loc=loc, scale=scale)
    return pdf
