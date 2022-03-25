import lovely_logger as logging
import numpy as np
import streamlit as st
from pandas import read_csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import matplotlib.pyplot as plt
import timeit
# from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
from shapely.geometry import LineString, Point
import plotly.graph_objs as go


def plot_NT(Frames, Nums, fps):
    logging.info("plot NT-curve")
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=["N-T"], x_title="time / s", y_title="N"
    )
    for i, frames in Frames.items():
        nums = Nums[i]
        if not frames:
            continue

        trace = go.Scatter(
            x=np.array(frames) / fps,
            y=nums,
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            marker=dict(size=1),
            line=dict(width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    # eps = 0.5
    # fig.update_xaxes(range=[xmin/fps - eps, xmax/fps + eps])
    # fig.update_yaxes(range=[ymin - eps, ymax + eps], autorange=False)
    fig.update_layout(
        width=500,
        height=500,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        autorange=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_flow(Frames, Nums, fps):
    logging.info("plot flow-curve")
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=["Flow"], x_title="time / s", y_title="J / 1/s"
    )
    for i, frames in Frames.items():
        nums = Nums[i]
        if not frames:
            continue

        times = np.array(frames) / fps
        trace = go.Scatter(
            x=times,
            y=nums / times,
            mode="lines",
            showlegend=True,
            name=f"ID: {i}",
            marker=dict(size=1),
            line=dict(width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    fig.update_layout(
        width=500,
        height=500,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        autorange=True,
    )
    st.plotly_chart(fig, use_container_width=False)


def plot_trajectories(data, geo_walls, transitions, min_x, max_x, min_y, max_y):
    fig = make_subplots(rows=1, cols=1)
    peds = np.unique(data[:, 0])
    for ped in peds:
        d = data[data[:, 0] == ped]
        c = d[:, -1]
        trace = go.Scatter(
            x=d[:, 2],
            y=d[:, 3],
            mode="lines",
            showlegend=False,
            name=f"{ped:0.0f}",
            marker=dict(size=1, color=c),
            line=dict(color="gray", width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    for gw in geo_walls.keys():
        trace = go.Scatter(
            x=geo_walls[gw][:, 0],
            y=geo_walls[gw][:, 1],
            showlegend=False,
            mode="lines",
            line=dict(color="black", width=2),
        )
        fig.append_trace(trace, row=1, col=1)

    for i, t in transitions.items():
        xm = np.sum(t[:, 0]) / 2
        ym = np.sum(t[:, 1]) / 2
        length = np.sqrt(np.diff(t[:, 0])**2 + np.diff(t[:, 1])**2)
        offset = 0.1 * length[0]
        logging.info(f"offsset transition {offset}")
        trace = go.Scatter(
            x=t[:, 0],
            y=t[:, 1],
            showlegend=False,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(color="black", size=5),
        )
        trace_text = go.Scatter(
            x=[xm + offset],
            y=[ym + offset],
            text=f"{i}",
            textposition="middle center",
            showlegend=False,
            mode="markers+text",
            marker=dict(color="red", size=0.1),
            textfont=dict(color="red", size=18),
        )
        fig.append_trace(trace, row=1, col=1)
        fig.append_trace(trace_text, row=1, col=1)

    eps = 1
    fig.update_layout(
        width=500,
        height=500,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        range=[min_y - eps, max_y + eps],
        # autorange=True,
    )
    fig.update_xaxes(
        #      #scaleanchor="y",
        #     # scaleratio=1,
        range=[min_x - eps, max_x + eps],
        #     autorange=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# def file_selector(folder_path="."):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox("Select a file", filenames)
#     return os.path.join(folder_path, selected_filename)


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0], _geometry_wall[gw][:, 1], color="white", lw=2)


def inv_weidmann(rho, v0=1.34, rho_max=5.4, gamma=1.913):
    inv_rho = np.empty_like(rho)
    mask = (rho == 0)
    inv_rho[mask] = v0
    inv_rho[~mask] = 1/rho[~mask]
    return v0*(1-np.exp(-gamma*(inv_rho - 1 / rho_max)))  # Eq. 6


def weidmann(v, v0=1.34, rho_max=5.4, gamma=1.913):
    v0 = np.max(v)
    v[v > v0] = v0
    s = 1 - v/v0
    x = -1 / gamma * np.log(s, out=np.zeros_like(s), where=(s != 0)) + 1 / rho_max
    return 1 / x


def get_fps(traj_file):
    fps = traj_file.split("#framerate:")[-1].split("\n")[0]
    try:
        fps = int(float(fps))
    except ValueError:
        logging.error(f"{fps} in header can not be converted to int")
        st.stop()

    return fps


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


def passing_frame(ped_data: np.array, line: LineString, fps: int) -> int:
    """Return frame of first time ped enters the line buffer

    Enlarge the line by eps, a constant that is dependent on fps
    eps = 1/fps * v0, v0 = 1.3 m/s

    :param ped_data: trajectories of ped
    :param line: transition
    :param fps: fps

    :returns: frame of entrance. Return negative number if ped did not pass trans

    """
    eps = 1 / fps * 1.3
    line_buffer = line.buffer(eps)
    for (frame, x, y) in ped_data[:, 1:4]:
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
                points = np.vstack([points, [vertex_x/cm2m, vertex_y/cm2m]])

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


def plot_profile(
    geominX,
    geomaxX,
    geominY,
    geomaxY,
    geometry_wall,
    dx,
    X,
    Y,
    Z,
    nframes,
    interpolation,
    cmap,
    label,
    title,
):
    """Plot profile + geometry for 3D data"""

    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    ret = stats.binned_statistic_2d(
        X,
        Y,
        Z,
        "sum",
        bins=[xbins, ybins],
    )
    prof = np.nan_to_num(ret.statistic.T)/nframes
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        prof,
        cmap=cmap,
        interpolation=interpolation,
        origin="lower",
        vmin=np.min(prof),
        vmax=np.max(prof),
        extent=[geominX, geomaxX, geominY, geomaxY],
    )
    plot_geometry(ax, geometry_wall)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, rotation=90, labelpad=15, fontsize=15)
    st.pyplot(fig)
    return prof


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
    return fwhm * 0.6005612  # np.sqrt(2) / (2 * np.sqrt(2 * np.log(2)))


def Gauss(x, a):
    """
    1 / (np.sqrt(np.pi) * a) * np.e ** (-x ** 2 / a ** 2)
    """

    return 1 / (1.7724538 * a) * np.e ** (-x ** 2 / a ** 2)


def densityField(x_dens, y_dens, a):
    rho_matrix_x = Gauss(x_dens, a)
    rho_matrix_y = Gauss(y_dens, a)
    rho_matrix = np.matmul(rho_matrix_x, np.transpose(rho_matrix_y))
    return rho_matrix.T


def xdensYdens(lattice_x, lattice_y, x_array, y_array):
    x_dens = np.add.outer(-x_array, lattice_x)
    y_dens = np.add.outer(-y_array, lattice_y)
    return x_dens, y_dens


def orderFieldPlot(
        geominX,
        geomaxX,
        geominY,
        geomaxY,
        geometry_wall,
        dx,
        nframes,
        X,
        Y,
        interpolation,
        cmap,
        label,
        title,):

    fig, ax = plt.subplots(1, 1)
    xbins = np.arange(geominX, geomaxX + dx, dx)
    ybins = np.arange(geominY, geomaxY + dx, dx)
    x_dens, y_dens = xdensYdens(X, Y, xbins, ybins)
    a = widthOfGaußian(0.3)
    rho_matrix = densityField(x_dens, y_dens, a)/nframes
    z_min, z_max = rho_matrix.min(), rho_matrix.max()
    im = ax.imshow(
        rho_matrix,
        cmap=cmap,
        interpolation=interpolation,
        origin="lower",
        vmin=z_min,
        vmax=z_max,
        extent=[geominX, geomaxX, geominY, geomaxY],
    )
    plot_geometry(ax, geometry_wall)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, rotation=90, labelpad=15, fontsize=15)
    st.pyplot(fig)
    return rho_matrix


def plot_profile_velocity(
        geominX,
        geomaxX,
        geominY,
        geomaxY,
        geometry_wall,
        density,
        interpolation,
        cmap,
        label,
        title,):

    fig, ax = plt.subplots(1, 1)
    speed = inv_weidmann(density)
    z_min, z_max = speed.min(), speed.max()
    im = ax.imshow(
        speed,
        cmap=cmap,
        interpolation=interpolation,
        origin="lower",
        vmin=z_min,
        vmax=z_max,
        extent=[geominX, geomaxX, geominY, geomaxY],
    )
    plot_geometry(ax, geometry_wall)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.3)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(label, rotation=90, labelpad=15, fontsize=15)
    st.pyplot(fig)
    return speed
