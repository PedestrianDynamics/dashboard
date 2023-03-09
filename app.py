import datetime as dt
import os
import timeit
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import lovely_logger as logging
import numpy as np
import streamlit as st
from hydralit import HydraApp
from apps import (
    about,
    dv_time_series,
    jams,
    loader,
    neighbors,
    profiles,
    rset,
    stats,
    time_series,
    trajectories,
)

import doc
import Utilities
import data_structure

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


def set_state_variables():
    if "bg_img" not in st.session_state:
        st.session_state.bg_img = None

    if "scale" not in st.session_state:
        st.session_state.scale = 0.5

    if "dpi" not in st.session_state:
        st.session_state.dpi = 100

    if "img_height" not in st.session_state:
        st.session_state.img_height = 100

    if "img_width" not in st.session_state:
        st.session_state.img_width = 100

    if "old_data" not in st.session_state:
        st.session_state.old_data = ""

    if "data" not in st.session_state:
        st.session_state.data = np.array([])

    if "orig_data" not in st.session_state:
        st.session_state.orig_data = np.array([])

    if "geometry_data" not in st.session_state:
        st.session_state.geometry_data = ""

    if "geominX" not in st.session_state:
        st.session_state.geominX = -10

    if "geomaxX" not in st.session_state:
        st.session_state.geomaxX = 10

    if "geominY" not in st.session_state:
        st.session_state.geominY = -10

    if "geomaxY" not in st.session_state:
        st.session_state.geomaxY = 10

    if "geometry_wall" not in st.session_state:
        st.session_state.geometry_wall = 10

    if "transitions" not in st.session_state:
        st.session_state.transitions = []

    if "fps" not in st.session_state:
        st.session_state.fps = 16

    if "speed_index" not in st.session_state:
        st.session_state.speed_index = -1

    if "header_traj" not in st.session_state:
        st.session_state.header_traj = ""

    if "density" not in st.session_state:
        st.session_state.density = []

    if "tstats" not in st.session_state:
        st.session_state.tstats = defaultdict(list)

    if "cum_num" not in st.session_state:
        st.session_state.cum_num = {}

    if "unit" not in st.session_state:
        st.session_state.unit = "m"

    if "df" not in st.session_state:
        st.session_state.df = 12

    if "xpos" not in st.session_state:
        st.session_state.xpos = None

    if "ypos" not in st.session_state:
        st.session_state.ypos = None

    if "lm" not in st.session_state:
        st.session_state.lm = None

    if "example_downloaded" not in st.session_state:
        st.session_state.example_downloaded = {}


def main():
    time_start = timeit.default_timer()

    set_state_variables()
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/PedestrianDynamics/dashboard"
    repo_name = f"[![Repo]({gh})]({repo})"
    st.sidebar.image(f"{ROOT_DIR}/figs/dashboard_logo.png", use_column_width=True)
    c1, c2 = st.sidebar.columns((1.2, 0.5))
    c2.markdown(repo_name, unsafe_allow_html=True)
    c1.write(
        "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7697604.svg)](https://doi.org/10.5281/zenodo.7697604)"
    )
    # c1.write( "[![Star](https://img.shields.io/github/stars/PedestrianDynamics/dashboard.svg?logo=github&style=social)](https://gitHub.com/PedestrianDynamics/dashboard)"
    # )

    from_examples = st.sidebar.selectbox(
        "📂 Select example",
        ["None"] + list(Utilities.examples.keys()),
        help="Select example. If files are uploaded (below), then this selection is invalidated",
    )
    c1, c2 = st.sidebar.columns((1, 1))
    trajectory_file = c1.file_uploader(
        "🚶 🚶‍♀️ Trajectory file ",
        type=["txt"],
        help="Load trajectory file",
    )
    # st.sidebar.markdown("-------")
    geometry_file = c2.file_uploader(
        "🏠 Geometry file ",
        type=["xml"],
        help="Load geometry file",
    )
    st.sidebar.markdown("-------")
    unit_pl = st.sidebar.empty()

    msg_status = st.sidebar.empty()
    disable_NT_flow = False
    files = data_structure.data_files(trajectory_file, geometry_file, from_examples)
    if files.got_traj_data:
        try:
            string_data = files.process_traj_file()
            if string_data != st.session_state.old_data:
                st.session_state.old_data = string_data
                new_data = True
                logging.info("Loading new trajectory data")
            else:
                logging.info("Trajectory data existing")
                new_data = False

            group_index = Utilities.get_index_group(string_data)
            if Utilities.detect_jpscore(string_data):
                how_speed = "from simulation"
                st.write(
                    "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
                    unsafe_allow_html=True,
                )
                df = 10
            else:
                how_speed = "from experiment"
                df = 10

            if new_data:
                with Utilities.profile("Load trajectories"):
                    data = files.get_data()
                    st.session_state.orig_data = np.copy(data)
                    fps = Utilities.get_fps(string_data)
                    speed_index = Utilities.get_speed_index(string_data)
                    header_traj = Utilities.get_header(string_data)
                    logging.info(f"Speed index: {speed_index}")
                    if speed_index == -1:  # data.shape[1] < 10:  # extend data
                        data = Utilities.compute_speed_and_angle(
                            data, fps, st.session_state.df
                        )

                    unit = Utilities.get_unit(string_data)
                    st.session_state.unit = unit
                    st.session_state.data = np.copy(data)
                    st.session_state.fps = fps
                    st.session_state.speed_index = speed_index
                    st.session_state.header_traj = header_traj
                    st.session_state.bg_img = None
                    logging.info("Done loading trajectories")

            else:
                with Utilities.profile("Second init"):
                    data = np.copy(st.session_state.data)
                    fps = st.session_state.fps
                    unit = st.session_state.unit
                    speed_index = st.session_state.speed_index
                    header_traj = st.session_state.header_traj
                    logging.info(
                        f"Second init of trajectories {st.session_state.data.shape}"
                    )

            if unit not in ["cm", "m"]:
                unit = unit_pl.radio(
                    "What is the unit of the trajectories?",
                    ["cm", "m"],
                    help="Choose the unit of the original trajectories. Data in the app will be converted to meter",
                )
                st.write(
                    "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
                    unsafe_allow_html=True,
                )
                st.sidebar.markdown("-------")

            logging.info(f"fps = {fps}")
        except Exception as e:
            msg_status.error(
                f"""Problem by initialising the trajectory data.
                Error: {e}"""
            )

            st.stop()

        try:
            geo_string_data = files.process_geo_file()
            if geo_string_data != st.session_state.geometry_data:
                with Utilities.profile("Load geometry:"):
                    # new_geometry = True
                    st.session_state.geometry_data = geo_string_data
                    geo_xml = files.read_geo_data()

                    logging.info("Geometry parsed successfully")
                    geometry_wall = Utilities.read_subroom_walls(geo_xml, unit="m")
                    logging.info("Got geometry walls successfully")
                    transitions = Utilities.get_transitions(geo_xml, unit="m")
                    logging.info("Got geometry transitions successfully")
                    measurement_lines = Utilities.get_measurement_lines(
                        geo_xml, unit="m"
                    )
                    logging.info("Got geometry measurement_lines successfully")
                    # todo: check if ids of transitions and measurement_lines are unique
                    transitions.update(measurement_lines)
                    logging.info("Get geo_limits")
                    geominX, geomaxX, geominY, geomaxY = Utilities.geo_limits(
                        geo_xml, unit="m"
                    )
                    st.session_state.geominX = geominX
                    st.session_state.geomaxX = geomaxX
                    st.session_state.geominY = geominY
                    st.session_state.geomaxY = geomaxY
                    st.session_state.transitions = deepcopy(transitions)
                    st.session_state.geometry_wall = deepcopy(geometry_wall)
                    st.session_state.xpos = None
                    st.session_state.ypos = None
                    st.session_state.lm = None
                    logging.info(
                        f"GeometrySize: X: ({geominX:.2f},{geomaxX:.2f}), Y: ({geominY:.2f},{geomaxY:.2f})"
                    )
            else:
                with Utilities.profile("Second geome init"):
                    # new_geometry = False
                    geominX = st.session_state.geominX
                    geomaxX = st.session_state.geomaxX
                    geominY = st.session_state.geominY
                    geomaxY = st.session_state.geomaxY
                    geometry_wall = deepcopy(st.session_state.geometry_wall)
                    transitions = deepcopy(st.session_state.transitions)

            # select all per default
            if transitions:
                default = list(transitions.keys())  # choose this transition by default
            else:
                default = []
                disable_NT_flow = True

        except Exception as e:
            msg_status.error(
                f"""Problem by initialising the geometry data.
                Error: {e}"""
            )
            st.stop()

        if unit == "cm":
            data[:, 2:4] /= 100
            data[:, st.session_state.speed_index] /= 100
            files.get_data_df()["X"] /= 100
            files.get_data_df()["Y"] /= 100
            files.get_data_df()["Z"] /= 100
            geominX /= 100
            geomaxX /= 100
            geominY /= 100
            geomaxY /= 100
            geometry_wall = {k: geometry_wall[k] / 100 for k in geometry_wall}
            transitions = {k: transitions[k] / 100 for k in transitions}
            logging.info(
                f"CM GeometrySize: X: ({geominX:.2f},{geomaxX:.2f}), Y: ({geominY:.2f},{geomaxY:.2f})"
            )

        if how_speed == "from simulation":
            logging.info("speed by simulation")
            Utilities.check_shape_and_stop(data.shape[1], how_speed)
        else:
            logging.info("speed by trajectory")
            if df != st.session_state.df:
                data = Utilities.compute_speed_and_angle(
                    st.session_state.orig_data, fps, df
                )
                st.session_state.data = np.copy(data)
                st.session_state.df = df
                if unit == "cm":
                    data[:, 2:4] /= 100
                    data[:, st.session_state.speed_index] /= 100

        pl.empty()
        app.add_loader_app(loader.MyLoadingApp())
        app.add_loader_app(loader.MyLoadingApp())
        app.add_app(
            "Summary", icon="🔢", app=stats.StatClass(data, unit, fps, header_traj)
        )
        app.add_app(
            "Trajectories",
            icon="👫🏻",
            app=trajectories.TrajClass(
                data,
                files.get_data_df(),
                how_speed,
                geometry_wall,
                transitions,
                geominX,
                geomaxX,
                geominY,
                geomaxY,
                fps,
            ),
        )
        app.add_app("Jam", icon="🐌 ", app=jams.JamClass(data, fps))
        name = files.traj_name
        # if traj_from_upload:
        #     name = trajectory_file.name.split(".txt")[0]
        # else:
        #     name = trajectory_file_d.split(".txt")[0]

        app.add_app(
            "Statistics",
            icon="📉 ",
            app=time_series.TimeSeriesClass(
                data, disable_NT_flow, transitions, default, fps, name, group_index
            ),
        )
        app.add_app(
            "Profiles",
            icon="🟡",
            app=profiles.ProfileClass(
                data, how_speed, geometry_wall, geominX, geomaxX, geominY, geomaxY, fps
            ),
        )
        app.add_app(
            "Time series",
            icon="🟠",
            app=dv_time_series.dvTimeSeriesClass(
                "Time series",
                data,
                how_speed,
                geometry_wall,
                geominX,
                geomaxX,
                geominY,
                geomaxY,
                fps,
                new_data,
            ),
        )

        app.add_app(
            "RSET",
            icon="🔵",
            app=rset.RsetClass(
                data, geominX, geomaxX, geominY, geomaxY, geometry_wall, fps
            ),
        )
        app.add_app(
            "Neighbors",
            icon="👥",
            app=neighbors.NeighborsClass(
                data, geominX, geomaxX, geominY, geomaxY, geometry_wall
            ),
        )
        # Add new tabs here
        # ----
        #
        app.add_app("About", icon="ℹ️", app=about.AboutClass())
        app.run()
        c1, c2 = st.columns((1, 1))

    time_end = timeit.default_timer()
    msg_time = Utilities.get_time(time_end - time_start)
    logging.info(f":clock8: Finished in {msg_time}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="JuPedSim-Analytics",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/jupedsim/jpscore",
            "Report a bug": "https://github.com/jupedsim/jpscore/issues",
            "About": "Open source framework for simulating, analyzing and visualizing pedestrian dynamics",
        },
    )

    # st.header(":information_source: Analytics dashboard")
    over_theme = {"txc_inactive": "#FFFFFF"}
    app = HydraApp(
        title="JuPedSim - Dashboard",
        favicon="🐙",
        navbar_animation=True,
        navbar_sticky=True,
        navbar_theme=over_theme,
    )

    # st.header("Dashboard")
    global pl
    pl = st.empty()
    with pl:
        doc.docs()

    with Utilities.profile("Main"):
        main()
