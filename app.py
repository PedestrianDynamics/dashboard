import datetime as dt
import os
from io import StringIO
from pathlib import Path
from xml.dom.minidom import parse, parseString
from collections import defaultdict
import lovely_logger as logging
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import read_csv
from plotly.subplots import make_subplots
from scipy import stats
from shapely.geometry import LineString
import Utilities

path = Path(__file__)
ROOT_DIR = path.parent.absolute()
home_path = str(Path.home())


@st.cache
def init_logger():
    T = dt.datetime.now()
    logging.info(f"init_logger at {T}")
    name = f"tmp_{T.year}-{T.month:02}-{T.day:02}_{T.hour:02}-{T.minute:02}-{T.second:02}.log"
    logfile = os.path.join(ROOT_DIR, name)
    logging.FILE_FORMAT = "[%(asctime)s] [%(levelname)-8s] - %(message)s"
    logging.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    logging.init(logfile, to_console=False)

    return logfile


st.set_page_config(
    page_title="JuPedSim",
    page_icon=":large_blue_circle:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/jupedsim/jpscore",
        "Report a bug": "https://github.com/jupedsim/jpscore/issues",
        "About": "Open source framework for simulating, analyzing and visualizing pedestrian dynamics",
    },
)


def file_selector(folder_path="."):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)


def plot_geometry(ax, _geometry_wall):
    for gw in _geometry_wall.keys():
        ax.plot(_geometry_wall[gw][:, 0],
                _geometry_wall[gw][:, 1],
                color="white",
                lw=2)


def weidmann(v, v0=1.34, rho_max=5.4, gamma=1.913):
    x = -1 / gamma * (np.log(1 - v / v0) + 1 / rho_max)
    return 1 / x


# https://plotly.com/python/heatmaps/
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
            line=dict(color="black", width=1),
        )
        fig.append_trace(trace, row=1, col=1)

    
    for i, t in transitions.items():
        trace = go.Scatter(
            x=t[:, 0],
            y=t[:, 1],
            showlegend=False,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(color="red", size=8),
        )
        trace_text = go.Scatter(
            x=[np.sum(t[:, 0])/2],
            y=[np.sum(t[:, 1])/2],
            text=f"ID: {i}",
            textposition='middle right',
            showlegend=False,
            mode="markers+text",
            line=dict(color="red", width=3),
            marker=dict(color="red", size=2),
            textfont=dict(color="red", size=18),
        )
        fig.append_trace(trace, row=1, col=1)
        fig.append_trace(trace_text, row=1, col=1)

    eps = 1
    fig.update_xaxes(range=[min_x - eps, max_x + eps])
    fig.update_yaxes(range=[min_y - eps, max_y + eps], autorange=False)

    st.plotly_chart(fig, use_container_width=True)


def plot_NT(Frames, Nums, fps):
    fig = make_subplots(rows=1, cols=1, x_title="time / s", y_title="N")
    for i, frames in Frames.items():
        nums = Nums[i]
        if not frames:
            continue

        trace = go.Scatter(
            x=np.array(frames)/fps,
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

    st.plotly_chart(fig, use_container_width=True)


def get_geometry_file(traj_file):
    return traj_file.split("geometry:")[-1].split("\n")[0].strip()


def set_state_variables():
    if "old_configs" not in st.session_state:
        st.session_state.old_configs = ""


def read_trajectory(input_file):
    data = read_csv(input_file,
                    sep=r"\s+",
                    dtype=np.float64,
                    comment="#").values

    return data


if __name__ == "__main__":
    set_state_variables()
    st.sidebar.image("jupedsim.png", use_column_width=True)
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/chraibi/jupedsim-dashboard"
    repo_name = f"[![Repo]({gh})]({repo})"
    st.sidebar.markdown(repo_name, unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns((1, 1))
    trajectory_file = c1.file_uploader(
        "📙 Trajectory file ",
        type=["txt"],
        help="Load trajectory file",
    )
    #st.sidebar.markdown("-------")
    geometry_file = c2.file_uploader(
        "🏠 Geometry file ",
        type=["xml"],
        help="Load geometry file",
    )
    st.sidebar.markdown("-------")
    c1, c2 = st.sidebar.columns((1, 1))
    choose_trajectories = c1.checkbox(
        "Trajectories", help="Plot trajectories", key="Traj"
    )
    choose_transitions = c2.checkbox(
        "Transitions", help="Show transittions", key="Tran"
    )
    c1, c2 = st.sidebar.columns((1, 1))
    choose_dprofile = c1.checkbox(
        "Density profiles", help="Plot density profiles", key="dProfile"
    )
    choose_vprofile = c2.checkbox(
        "Velocity profiles", help="Plot velocity profiles", key="vProfile"
    )
    dx = st.sidebar.slider("Step", 0.01, 0.5, 0.2, help="Space discretization")
    st.sidebar.markdown("-------")
    choose_NT = st.sidebar.checkbox("N-T diagram", help="Plot N-t curve", key="NT")
    msg_status = st.sidebar.empty()
    if trajectory_file and geometry_file:
        logging.info(f">> {trajectory_file.name}")
        logging.info(f">> {geometry_file.name}")
        try:
            data = read_trajectory(trajectory_file)
            stringio = StringIO(trajectory_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            fps = Utilities.get_fps(string_data)
            peds = np.unique(data[:, 0])
            st.markdown(":bar_chart: Statistics")
            msg = f"""
            Frames per second: {fps}\n
            Agents: {len(peds)}\n
            Evac-time: {np.max(data[:, 1])/fps} seconds
            """
            st.info(msg)
            logging.info(f"fps = {fps}")
        except Exception as e:
            msg_status.error(
                f"""Can't parse trajectory file.
                Error: {e}"""
            )
            st.stop()

        try:
            parse_geometry_file = get_geometry_file(string_data)
            logging.info(f"Geometry: <{geometry_file}>")
            # Read Geometry file
            if parse_geometry_file != geometry_file.name:
                st.error(f"Mismatched geometry files. Parsed {parse_geometry_file}. Uploaded {geometry_file.name}")
                st.stop()
            
            stringio = StringIO(geometry_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            file_data = geometry_file.read()
    
            geo_xml = parseString(geometry_file.getvalue())
            geometry_wall = Utilities.read_subroom_walls(geo_xml)
            transitions = Utilities.get_transitions(geo_xml)
            selected_transitions = st.sidebar.multiselect(
                'Select transition',
                transitions.keys(),
                help="Transition to calculate N-T. Can select multiple transitions",
                )
            logging.info(transitions)
            geominX, geomaxX, geominY, geomaxY = Utilities.geo_limits(geo_xml)

            print(
                "GeometrySize: X: ({:.2f},{:2f}), Y: ({:.2f},{:2f})".format(
                    geominX, geomaxX, geominY, geomaxY
                )
            )

        except Exception as e:
            msg_status.error(
                f"""Can't parse geometry file.
                Error: {e}"""
            )
            st.stop()


        if choose_trajectories:
            logging.info("plotting trajectories")
            if choose_transitions:
                plot_trajectories(data, geometry_wall, transitions, geominX, geomaxX, geominY, geomaxY)
            else:
                plot_trajectories(data, geometry_wall, {}, geominX, geomaxX, geominY, geomaxY)

        if choose_dprofile:
            logging.info("plotting density profile")
            xbins = np.arange(geominX, geomaxX + dx, dx)
            ybins = np.arange(geominY, geomaxY + dx, dx)

            ret2 = stats.binned_statistic_2d(
                data[:, 2],
                data[:, 3],
                weidmann(data[:, 9]),
                "mean",
                bins=[xbins, ybins],
            )
            prof2 = np.nan_to_num(ret2.statistic.T)

            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(
                prof2,
                cmap=cm.jet,
                interpolation="bicubic",
                origin="lower",
                vmin=0,
                vmax=5,  # np.max(data[:, 9]),
                extent=[geominX, geomaxX, geominY, geomaxY],
            )
            plot_geometry(ax, geometry_wall)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.5%", pad=0.3)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(r"$\rho\; / 1/m^2$", rotation=90, labelpad=15, fontsize=15)

            st.pyplot(fig)

        if choose_vprofile:
            logging.info("plotting velocity profile")
            xbins = np.arange(geominX, geomaxX + dx, dx)
            ybins = np.arange(geominY, geomaxY + dx, dx)

            ret = stats.binned_statistic_2d(
                data[:, 2],
                data[:, 3],
                data[:, 9],
                "mean",
                bins=[xbins, ybins]
            )
            prof = np.nan_to_num(ret.statistic.T)
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(
                prof,
                cmap=cm.jet.reversed(),
                interpolation="bicubic",
                origin="lower",
                vmin=0,
                vmax=1.34,  # np.max(data[:, 9]),
                extent=[geominX, geomaxX, geominY, geomaxY],
            )

            plot_geometry(ax, geometry_wall)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.5%", pad=0.3)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(r"$v\; / m/s$", rotation=90, labelpad=15, fontsize=15)

            st.pyplot(fig)

        if choose_NT:
            peds = np.unique(data)
            stats = defaultdict(list)
            cum_num = {}
            for i, t in transitions.items():
                if i in selected_transitions:
                    line = LineString(t)
                    for ped in peds:
                        ped_data = data[data[:, 0] == ped]
                        frame = Utilities.passing_frame(ped_data, line, fps)
                        if frame >= 0:
                            stats[i].append(frame)

                stats[i].sort()
                cum_num[i] = np.cumsum(np.ones(len(stats[i])))
                
            plot_NT(stats, cum_num, fps)
            logging.info(stats)
            logging.info(cum_num)




                    
