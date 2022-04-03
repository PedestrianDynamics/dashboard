import contextlib
import time
from collections import defaultdict

import lovely_logger as logging
import numpy as np
import pandas as pd
import streamlit as st
from pandas import read_csv
from scipy import stats
from shapely.geometry import LineString, Point


@contextlib.contextmanager
def profile(name):
    start_time = time.time()
    yield  # <-- your code will execute here
    total_time = time.time() - start_time
    print("%s: %.4f ms" % (name, total_time * 1000.0))


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
    x = -1 / gamma * np.log(s, out=np.zeros_like(s), where=(s != 0)) + 1 / rho_max
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

    measurement_lines = {}
    for _, t_elem in enumerate(xml_doc.getElementsByTagName("area_L")):
        Id = t_elem.getAttribute("id")
        logging.info(f"Measurement {Id}")
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


def passing_frame(
    ped_data: np.array, line: LineString, fps: int, max_distance: float
) -> int:
    """Return frame of first time ped enters the line buffer

    Enlarge the line by eps, a constant that is dependent on fps
    eps = 1/fps * v0, v0 = 1.3 m/s

    :param ped_data: trajectories of ped
    :param line: transition
    :param fps: fps
    : param max_distance: an arbitrary distance to the line

    :returns: frame of entrance. Return negative number if ped did not pass trans

    """
    eps = 1 / fps * 1.3
    line_buffer = line.buffer(eps)
    p = ped_data[np.abs(ped_data[:, 2] - line.centroid.x) < max_distance]
    for (frame, x, y) in p[:, 1:4]:
        if Point(x, y).within(line_buffer):
            return frame

    return -1


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
    for o_num, o_elem in enumerate(xml_doc.getElementsByTagName("obstacle")):
        N_polygon = len(o_elem.getElementsByTagName("polygon"))
        if N_polygon == 1:
            pass
        else:
            points = np.zeros((0, 2))

        for p_num, p_elem in enumerate(o_elem.getElementsByTagName("polygon")):
            for v_num, v_elem in enumerate(p_elem.getElementsByTagName("vertex")):
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
        sort_tups = sorted(
            [(i, j, k) for i, j, k in zip(x, y, angles)], key=lambda t: t[2]
        )
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
            if True or p_elem.getAttribute("caption") == "wall":
                n_wall = n_wall + 1
                n_vertex = len(p_elem.getElementsByTagName("vertex"))
                vertex_array = np.zeros((n_vertex, 2))
                for v_num, _ in enumerate(p_elem.getElementsByTagName("vertex")):
                    vertex_array[v_num, 0] = (
                        p_elem.getElementsByTagName("vertex")[v_num]
                        .attributes["px"]
                        .value
                    )
                    vertex_array[v_num, 1] = (
                        p_elem.getElementsByTagName("vertex")[v_num]
                        .attributes["py"]
                        .value
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
                f"""The number of frames used to calculate the speed {df}
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


# def compute_agent_speed_and_angle(agent, fps, df=10):
#     """Calculates the speed and the angle from the trajectory points.

#     """

#     traj = agent[:, 2:4]
#     size = traj.shape[0]
#     speed = np.ones(size)
#     angle = np.zeros(size)
#     if size < df:
#         logging.warning(
#             f"""The number of frames used to calculate the speed {df}
#             exceeds the total amount of frames ({size}) in this trajectory."""
#         )
#         st.stop()

#     delta = traj[df:, :] - traj[: size - df, :]
#     delta_x = delta[:, 0]
#     delta_y = delta[:, 1]
#     delta_square = np.square(delta)
#     delta_x_square = delta_square[:, 0]
#     delta_y_square = delta_square[:, 1]
#     angle[: size - df] = np.arctan2(delta_y, delta_x) * 180 / np.pi
#     s = np.sqrt(delta_x_square + delta_y_square)
#     speed[: size - df] = s / df * fps
#     speed[size - df :] = speed[size - df - 1]
#     angle[size - df :] = angle[size - df - 1]

#     return speed, angle


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
    speeds = np.array([])

    for agent in agents:
        ped = data[data[:, 0] == agent]
        traj = ped[:, 2:4]
        size = traj.shape[0]
        speed = np.ones(size)
        angle = np.zeros(size)

        if size < df:
            logging.warning(
                f"""The number of frames used to calculate the speed {df}
                exceeds the total amount of frames ({size}) in this trajectory."""
            )
            st.stop()

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


def compute_agent_speed_and_angle(agent, fps, df=10):
    """Calculates the speed and the angle from the trajectory points."""

    traj = agent[:, 2:4]
    size = traj.shape[0]
    speed = np.ones(size)
    angle = np.zeros(size)
    if size < df:
        logging.warning(
            f"""The number of frames used to calculate the speed {df}
            exceeds the total amount of frames ({size}) in this trajectory."""
        )
        st.stop()

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

    return speed, angle


def calculate_speed_average(
    geominX, geomaxX, geominY, geomaxY, dx, nframes, X, Y, speed
):
    """Calculate speed average over time"""
    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    ret = stats.binned_statistic_2d(
        X,
        Y,
        speed,
        "mean",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T)


def calculate_density_average_weidmann(
    geominX, geomaxX, geominY, geomaxY, dx, nframes, X, Y, speed
):
    """Calculate density using Weidmann(speed)"""
    density = inv_weidmann(speed)
    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    ret = stats.binned_statistic_2d(
        X,
        Y,
        density,
        "mean",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T)  # / nframes


def calculate_density_average_classic(
    geominX, geomaxX, geominY, geomaxY, dx, nframes, X, Y
):
    """Calculate classical method

    Density = mean_time(N/A_i)
    """

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    area = dx * dx
    ret = stats.binned_statistic_2d(
        X,
        Y,
        None,
        "count",
        bins=[xbins, ybins],
    )

    return np.nan_to_num(ret.statistic.T) / nframes / area


def calculate_density_frame_classic(geominX, geomaxX, geominY, geomaxY, dx, X, Y):
    """Calculate classical method

    Density = mean_time(N/A_i)
    """

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    area = dx * dx
    ret = stats.binned_statistic_2d(
        X,
        Y,
        None,
        "count",
        bins=[xbins, ybins],
    )
    return np.nan_to_num(ret.statistic.T) / area


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


def widthOfGaußian(fwhm):
    """np.sqrt(2) / (2 * np.sqrt(2 * np.log(2)))"""

    return fwhm * 0.6005612


def Gauss(x, a):
    """1 / (np.sqrt(np.pi) * a) * np.e ** (-x ** 2 / a ** 2)"""

    return 1 / (1.7724538 * a) * np.e ** (-(x**2) / a**2)


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
    geominX, geomaxX, geominY, geomaxY, dx, nframes, width, X, Y
):
    """
    Calculate density using Gauss method
    """

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    x_dens, y_dens = xdensYdens(X, Y, xbins, ybins)
    a = widthOfGaußian(width)
    rho_matrix = densityField(x_dens, y_dens, a) / nframes
    return rho_matrix


def jam_frames(data, jam_speed):
    """Definition of jam

    return data in jam
    """

    jam_data = data[data[:, st.session_state.speed_index] <= jam_speed]
    return np.unique(jam_data[:, 1])


def consecutive_chunks(data1d, fps, frame_margin):
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

    logging.info(f"Jam chunks {chunks}. From-To: {ret}, Max = {np.max(chunks)}")
    return chunks, np.array(ret)


def jam_waiting_time(
        jam_data: np.array,
        jam_min_duration: int,
        fps: int,
        precision
):
    """Return a list of pid and its max_time in jam

    return a 2D array [ped, waiting_time]
    """
    waiting_times = []
    peds = np.unique(jam_data[:, 0]).astype(int)
    for ped in peds:
        jam_data_ped = jam_data[jam_data[:, 0] == ped]
        jam_times = consecutive_chunks(jam_data_ped[:, 1], fps, precision)
        max_waiting_time = np.max(jam_times)
        if max_waiting_time >= jam_min_duration:
            waiting_times.append([ped, max_waiting_time / fps])

    return np.array(waiting_times)


def jam_lifetime(data: np.array,
                 jam_frames,
                 jam_min_agents: int,
                 fps: int,
                 precision: int):
    """Lifespane of a Jam and how many pedestrian in chunck"""

    # frames = data[:, 1]
    lifetime = []
    # frame, num peds in jam. Using only the first,
    # since I dont know yet how to use the second
    # Ignore the first frames, where agents start from 0 (so in jam)
    for frame in jam_frames:
        if frame in jam_frames:
            d = data[data[:, 1] == frame]
            num_ped_in_jam = len(d[:, 0])
            if num_ped_in_jam >= jam_min_agents:
                lifetime.append([frame, num_ped_in_jam])

    lifetime = np.array(lifetime)

    if not lifetime.size:
        return np.array([]), np.array([]), 0, np.array([])

    chuncks, ret = consecutive_chunks(np.array(lifetime)[:, 0], fps, precision)
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

    """
    tstats = defaultdict(list)
    cum_num = {}
    msg = ""
    trans_used = {}

    peds = np.unique(data[:, 0]).astype(int)
    with st.spinner("Processing ..."):
        max_len = -1
        # longest array. Needed to stack arrays and save them in file
        for i, t in transitions.items():
            trans_used[i] = False
            if i in selected_transitions:
                line = LineString(t)
                len_line = line.length
                for ped in peds:
                    ped_data = data[data[:, 0] == ped]
                    frame = passing_frame(ped_data, line, fps, len_line)
                    if frame >= 0:
                        tstats[i].append([ped, frame])
                        trans_used[i] = True

                if trans_used[i]:
                    tstats[i] = np.array(tstats[i])
                    tstats[i] = tstats[i][tstats[i][:, 1].argsort()]  # sort by frame
                    arrivals = tstats[i][:, 1]
                    cum_num[i] = np.cumsum(np.ones(len(arrivals)))
                    flow = (cum_num[i][-1] - 1) / (arrivals[-1] - arrivals[0]) * fps
                    with profile("rolling flow: "):
                        mean_flow, std_flow = rolling_flow(arrivals, fps, windows=100)

                    max_len = max(max_len, cum_num[i].size)
                    msg += f"Transition {i}: length {line.length:.2f}, flow: {flow:.2f} [1/s], rolling_flow: {mean_flow:.2} +- {std_flow:.3f} [1/s],  specific flow: {flow/line.length:.2f} [1/s/m] \n \n"
                else:
                    msg += (
                        f"Transition {i}: length {line.length:.2f}, flow: 0 [1/s] \n \n"
                    )

    return tstats, cum_num, trans_used, max_len, msg


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
