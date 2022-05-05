import datetime as dt
import os
from collections import defaultdict
from copy import deepcopy
from io import StringIO
from pathlib import Path
from xml.dom.minidom import parseString, parse

import lovely_logger as logging
import numpy as np
# import plotly.graph_objs as go
import streamlit as st
from shapely.geometry import LineString
import timeit
import doc
import plots
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


def set_state_variables():
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
    st.header(":information_source: Dashboard")
    info = st.expander("click to expand")
    with info:
        doc.docs()

    set_state_variables()
    st.sidebar.image("figs/jupedsim.png", use_column_width=True)
    gh = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    repo = "https://github.com/chraibi/jupedsim-dashboard"
    repo_name = f"[![Repo]({gh})]({repo})"
    st.sidebar.markdown(repo_name, unsafe_allow_html=True)    
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
    time_msg = st.sidebar.empty()
    st.sidebar.markdown("-------")
    unit_pl = st.sidebar.empty()
    st.sidebar.header("📉 Plot trajectories")
    px = st.sidebar.expander("Options", expanded=True)
    c1, c2 = px.columns((1, 1))
    choose_trajectories = c1.checkbox(
        "Trajectories", help="Plot trajectories", key="Traj", value=True,
    )
    if choose_trajectories:
        choose_transitions = c2.checkbox(
            "Transitions", help="Show transittions", value=True, key="Tran"
        )
        show_special_agent_stats = c1.checkbox(
            "Agent", help="Show speed/angle and trajectory of highlighted agent", value=False, key="SpecialAgent"
        )

    pl_select_special_agent = px.empty()
    pl_sample_trajectories = px.empty()
    # ---------------------------------
    st.sidebar.header("🔵 Speed")
    sx = st.sidebar.expander("Options")
    #how_speed_pl = sx.empty()
    #df_pl = sx.empty()
    # ---------------------------------
    st.sidebar.header("🔴 Heatmaps")
    prfx = st.sidebar.expander("Options")
    c1, c2 = prfx.columns((1, 1))
    choose_dprofile = c1.checkbox(
        "▶️ Show", help="Plot density and speed profiles", key="dProfile"
    )
    # choose_vprofile = c2.checkbox("Speed", help="Plot speed profile", key="vProfile")
    choose_d_method = prfx.radio(
        "Density method",
        ["Classical", "Gaussian", "Weidmann"],
        help="""
        How to calculate average of density over time and space""",
    )
    prfx.write(
         "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
         unsafe_allow_html=True,
    )
    if choose_d_method == "Gaussian":
        width = prfx.slider(
            "Width", 0.05, 1.0, 0.6, help="Width of Gaussian function"
        )

    dx = prfx.slider("Grid size", 0.1, 4.0, 1.0, step=0.2, help="Space discretization")
    # methods = ["nearest", "gaussian", "sinc", "bicubic", "mitchell", "bilinear"]
    methods = ["off", "on"]
    interpolation = prfx.radio(
        "Smooth", methods, help="Smoothen the heatmaps"
    )
    if interpolation == "off":
        interpolation = "false"
    else:
        interpolation = "best"
    # prfx.write(
    #      "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
    #      unsafe_allow_html=True,
    # )
    if choose_dprofile:
        st.sidebar.header("📈 Density and speed timeseries")
        svp = st.sidebar.expander("Options")
        choose_timeseries = svp.checkbox(
            "▶️ Show", help="Plot density and speed timeseries", key="timeseries"
        )
        c1, c2 = svp.columns((1, 1))
        posx_pl = svp.empty()
        posy_pl = svp.empty()
        side_pl = svp.empty()
        sample_pl = svp.empty()
    else:
        choose_timeseries = False

    st.sidebar.header("📊 Summary curves")
    pc = st.sidebar.expander("Options")
    c1, c2 = pc.columns((1, 1))
    # ----- Jam
    st.sidebar.header("🐌 Jam")
    jp = st.sidebar.expander("Options")
    choose_jam_duration = jp.checkbox(
        "▶️ Show",
        value=False,
        help="Plot change of the number of pedestrian in jam versus time",
        key="jam_duration",
    )
    jam_speed = jp.slider(
        "Min jam speed / m/s",
        0.1,
        1.0,
        0.5,
        help="An agent slower that this speed is in jam",
        key="jVmin",
    )
    min_jam_time = jp.slider(
        "Min jam duration / s",
        1,
        180,
        1,
        help="A jam lasts at least that long",
        key="jTmin",
    )
    min_jam_agents = jp.slider(
        "Min agents in jam",
        2,
        200,
        20,
        help="A jam has at least so many agents",
        key="jNmin",
    )

    msg_status = st.empty()
    disable_NT_flow = False
    if (trajectory_file and geometry_file) or from_examples != "None":
        traj_from_upload = True
        if not trajectory_file and not geometry_file:
            traj_from_upload = False
            selection = Utilities.selected_traj_geo(from_examples)
            name_selection = selection[0]
            trajectory_file_d = name_selection + ".txt"
            geometry_file_d = name_selection + ".xml"
            if name_selection not in st.session_state.example_downloaded:
                st.session_state.example_downloaded[name_selection] = True
                logging.info(f"Downloading selected {from_examples}")
                Utilities.download(selection[1], trajectory_file_d)
                Utilities.download(selection[2], geometry_file_d)
            else:
                logging.info(f"Using selected {from_examples}")

        else:
            logging.info(f">> {trajectory_file}")
            logging.info(f">> {geometry_file}")
            if trajectory_file is None:
                st.error(
                    "No trajectory file uploaded yet!"
                )
                st.stop()

            if geometry_file is None:
                st.error(
                    "No geometry file uploaded yet!"
                )
                st.stop()

        try:
            logging.info(f"Trajectory from upload: {traj_from_upload}")
            h = st.expander("Head of Trajectories (first 4 columns)", expanded=False)
            if traj_from_upload:
                stringio = StringIO(trajectory_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
            else:
                with open(trajectory_file_d, encoding="utf-8") as f:
                    string_data = f.read()

            if string_data != st.session_state.old_data:
                st.session_state.old_data = string_data
                new_data = True
                logging.info("Loading new trajectory data")
            else:
                logging.info("Trajectory data existing")
                new_data = False

            if Utilities.detect_jpscore(string_data):
                how_speed = sx.radio(
                    "source", ["from simulation", "from trajectory"]
                )
                st.write(
                    "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
                    unsafe_allow_html=True,
                )
                df = 10
            else:
                how_speed = "from experiment"
                df = sx.slider(
                    "df",
                    2,
                    50,
                    10,
                    help="how many frames to consider for calculating the speed",
                )

            if new_data:
                with Utilities.profile("Load trajectories"):
                    logging.info("Load trajectories ..")
                    if not traj_from_upload:
                        data = Utilities.read_trajectory(trajectory_file_d)
                    else:
                        data = Utilities.read_trajectory(trajectory_file)

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

            with h:
                with Utilities.profile("show_table"):
                    logging.info(f"show table with {data.shape}")
                    fig = plots.show_trajectories_table(data[:10, 0:5])
                    st.plotly_chart(fig, use_container_width=True)

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

            st.markdown("### :bar_chart: Summary of the data")
            stats_msg = st.empty()
            frames = np.unique(data[:, 1])
            peds = np.unique(data[:, 0]).astype(int)
            nagents = len(peds)
            msg = f"""
            Trajectory column names: {header_traj}\n
            Unit: {unit}\n
            Frames per second: {fps}\n
            Frames: {len(frames)} | First: {frames[0]:.0f} | Last: {frames[-1]:.0f}\n
            Agents: {nagents}\n
            Evac-time: {np.max(frames)/fps} [s]
            """
            stats_msg.info(msg)
            f = st.expander("Documentation: Speed (click to expand)")
            with f:
                doc.doc_speed()

            if nagents <= 10:
                special_agent = 5
            else:
                special_agent = peds[10]

            sample_trajectories = pl_sample_trajectories.number_input(
                "sample",
                min_value=1,
                max_value=int(np.max(frames * 0.2)),
                value=10,
                step=5,
                help="Sample rate of ploting trajectories and time series \
                (the lower the slower)",
                key="sample_traj",
            )
            # hack. I dont know why this is sometimes float
            sample_trajectories = int(sample_trajectories)
            plot_ped = pl_select_special_agent.number_input(
                "Highlight pedestrian",
                min_value=np.min(peds),
                max_value=np.max(peds),
                value=special_agent,
                step=10,
                help="Choose a pedestrian by id",
            )

            logging.info(f"fps = {fps}")
        except Exception as e:
            msg_status.error(
                f"""Problem by initialising the trajectory data.
                Error: {e}"""
            )
            st.stop()

        try:
            if traj_from_upload:
                parse_geometry_file = Utilities.get_geometry_file(string_data)
                logging.info(f"Geometry: <{geometry_file.name}>")
                # Read Geometry file
                if parse_geometry_file != geometry_file.name:
                    st.error(
                        f"Mismatched geometry files. Parsed {parse_geometry_file}. Uploaded {geometry_file.name}"
                    )
                    st.stop()

                geo_stringio = StringIO(geometry_file.getvalue().decode("utf-8"))
                geo_string_data = geo_stringio.read()
            else:
                with open(geometry_file_d, encoding="utf-8") as geometry_file_obj:
                    geo_string_data = geometry_file_obj.read()

            if geo_string_data != st.session_state.geometry_data:
                with Utilities.profile("Load geometry:"):
                    # new_geometry = True
                    st.session_state.geometry_data = geo_string_data
                    if traj_from_upload:
                        geo_xml = parseString(geometry_file.getvalue())
                    else:
                        geo_xml = parse(geometry_file_d)

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

            geominX /= 100
            geomaxX /= 100
            geominY /= 100
            geomaxY /= 100
            geometry_wall = {k: geometry_wall[k] / 100 for k in geometry_wall}
            transitions = {k: transitions[k] / 100 for k in transitions}
            logging.info(
                f"CM GeometrySize: X: ({geominX:.2f},{geomaxX:.2f}), Y: ({geominY:.2f},{geomaxY:.2f})"
            )

        NT_form = pc.form("plot-NT")
        with NT_form:
            choose_NT = c1.checkbox(
                "N-T",
                value=True,
                help="Plot N-t curve",
                key="NT",
                disabled=disable_NT_flow,
            )
            choose_flow = c2.checkbox(
                "Flow",
                value=True,
                help="Plot flow curve",
                key="Flow",
                disabled=disable_NT_flow,
            )
            choose_time_distance = c1.checkbox(
                "T-D",
                value=True,
                help="Plot Time-Distance to the first selected entrance",
                key="EvacT",
                disabled=disable_NT_flow,
            )
            choose_survival = c2.checkbox(
                "Survival",
                value=True,
                help="Plot survival function (clogging)",
                disabled=disable_NT_flow,
                key="Survival",
            )
            num_peds_TD = pc.number_input(
                "number pedestrians",
                min_value=1,
                max_value=len(peds),
                value=int(0.3 * len(peds)),
                step=1,
                help="number of pedestrians to show in T-D",
            )
            sample_TD = pc.number_input(
                "sample",
                min_value=1,
                max_value=int(0.1 * len(frames)),
                value=int(0.01 * len(frames)),
                step=1,
                help="sample rate in T-D",
            )

        selected_transitions = NT_form.multiselect(
            "Select transition",
            transitions.keys(),
            default,
            help="Transition to calculate N-T. Can select multiple transitions",
        )
        make_plots = NT_form.form_submit_button(label="▶️ Show")
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

        if choose_trajectories:
            agent = data[data[:, 0] == plot_ped]
            speed_agent = agent[:, st.session_state.speed_index]
            if how_speed == "from simulation":
                angle_agent = agent[:, 7]
            else:
                angle_agent = agent[:, -2]

            with Utilities.profile("plot_trajectories"):
                fig = plots.plot_trajectories(
                    data,
                    plot_ped,
                    speed_agent,
                    geometry_wall,
                    transitions,
                    geominX,
                    geomaxX,
                    geominY,
                    geomaxY,
                    choose_transitions,
                    sample_trajectories,
                )
                st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns((1, 1, 1))
            if show_special_agent_stats:
                with Utilities.profile("plot_agent_xy"):
                    fig = plots.plot_agent_xy(
                        agent[::sample_trajectories, 1],
                        agent[::sample_trajectories, 2],
                        agent[:, 3],
                        fps,
                    )
                    c1.plotly_chart(fig, use_container_width=True)

                with Utilities.profile("plot_agent_angle"):
                    fig = plots.plot_agent_angle(
                        plot_ped,
                        agent[::sample_trajectories, 1],
                        angle_agent[::sample_trajectories],
                        fps,
                    )
                    c2.plotly_chart(fig, use_container_width=True)

                with Utilities.profile("plot_agent_speed"):

                    fig = plots.plot_agent_speed(
                        plot_ped,
                        agent[::sample_trajectories, 1],
                        speed_agent[::sample_trajectories],
                        np.max(data[:, st.session_state.speed_index]),
                        fps,
                    )
                    c3.plotly_chart(fig, use_container_width=True)

        # choose_dprofile =
        choose_vprofile = True  # todo: not sure is I want to keep this option
        if choose_dprofile:
            if st.session_state.xpos is not None:  # no need to check ypos and lm
                xm = st.session_state.xpos
                ym = st.session_state.ypos
                lm = st.session_state.lm
            else:
                if st.session_state.transitions:
                    v = [x for x in transitions.values()]
                    t = v[0]
                    xm = np.sum(t[:, 0]) / 2
                    ym = np.sum(t[:, 1]) / 2
                else:
                    xm = 0.5 * (geominX + geomaxX)
                    ym = 0.5 * (geominY + geomaxY)

            xpos = posx_pl.number_input(
                "x-position of measurement area",
                min_value=float(geominX),
                max_value=float(geomaxX),
                value=xm,
                step=dx,
                format="%.1f",
                help="X-ccordinate of the center of the measurement square.",
            )
            ypos = posy_pl.number_input(
                "y-position of measurement area",
                min_value=float(geominY),
                max_value=float(geomaxY),
                value=ym,
                step=dx,
                format="%.1f",
                help="Y-ccordinate of the center of the measurement square.",
            )
            lm = side_pl.number_input(
                "side of square",
                min_value=0.2,
                max_value=5.0,
                value=1.0,
                step=0.5,
                format="%.1f",
                help="Length of the side of the measurement square.",
            )
            sample = sample_pl.number_input(
                "sample",
                min_value=1,
                max_value=int(np.max(frames * 0.2)),
                value=10,
                step=5,
                help="Sample rate of ploting time series \
                (the lower the slower)",
            )

            if xpos != st.session_state.xpos:
                st.session_state.xpos = xpos

            if ypos != st.session_state.ypos:
                st.session_state.ypos = ypos

            if lm != st.session_state.lm:
                st.session_state.lm = lm

        else:
            st.session_state.xpos = None
            st.session_state.ypos = None
            st.session_state.lm = None

        info_profile = st.expander(
            "Documentation: Density/Speed maps (click to expand)"
        )
        messages = st.empty()
        c1, c2 = st.columns((1, 1))
        dprofile_pl = c1.empty()
        vprofile_pl = c2.empty()
        
        info_rset = st.expander(
            "Documentation: RSET maps (click to expand)"
        )
        c1, c2 = st.columns((1, 1))
        rset_pl = c2.empty()
        rset2_pl = c1.empty()
        info_timeseries = st.expander(
            "Documentation: Density/Speed Time Series (click to expand)"
        )
        c1, c2 = st.columns((1, 1))
        #plot_timeseries_pl = c1.empty()
        dtimeseries_pl = c1.empty()
        vtimeseries_pl = c2.empty()

        with info_timeseries:
            doc.doc_timeseries()

        with info_profile:
            doc.doc_profile()

        with info_rset:
            doc.doc_RSET()
            
        if choose_dprofile:
            Utilities.check_shape_and_stop(data.shape[1], how_speed)
            msg = ""

            with st.spinner("Processing ..."):
                if choose_dprofile:
                    xbins = np.arange(geominX, geomaxX + dx, dx)
                    ybins = np.arange(geominY, geomaxY + dx, dx)
                    if choose_d_method == "Weidmann":
                        density_ret = Utilities.calculate_density_average_weidmann(
                            geominX,
                            geomaxX,
                            geominY,
                            geomaxY,
                            dx,
                            len(frames),
                            data[:, 2],
                            data[:, 3],
                            data[:, st.session_state.speed_index],
                        )
                        # time serie
                        if choose_timeseries:
                            density_time = []
                            for frame in frames[::sample]:
                                dframe = data[:, 1] == frame
                                x = data[dframe][:, 2]
                                y = data[dframe][:, 3]
                                speed_agent = data[dframe][
                                    :, st.session_state.speed_index
                                ]
                                dtime = Utilities.calculate_density_average_weidmann(
                                    st.session_state.xpos - st.session_state.lm / 2,
                                    st.session_state.xpos + st.session_state.lm / 2,
                                    st.session_state.ypos - st.session_state.lm / 2,
                                    st.session_state.ypos + st.session_state.lm / 2,
                                    st.session_state.lm,
                                    1,
                                    x,
                                    y,
                                    speed_agent,
                                )
                                density_time.append(dtime[0, 0])
                    elif choose_d_method == "Gaussian":
                        with Utilities.profile("density profile gauss"):
                            density_ret = Utilities.calculate_density_average_gauss(
                                geominX,
                                geomaxX,
                                geominY,
                                geomaxY,
                                dx,
                                len(frames),
                                width,
                                data[:, 2],
                                data[:, 3],
                            )
                        if choose_timeseries:
                            # time serie
                            with Utilities.profile("time series gauss"):
                                density_time = []
                                for frame in frames[::sample]:
                                    dframe = data[:, 1] == frame
                                    x = data[dframe][:, 2]
                                    y = data[dframe][:, 3]
                                    dtime = Utilities.calculate_density_average_gauss(
                                        st.session_state.xpos - st.session_state.lm / 2,
                                        st.session_state.xpos + st.session_state.lm / 2,
                                        st.session_state.ypos - st.session_state.lm / 2,
                                        st.session_state.ypos + st.session_state.lm / 2,
                                        st.session_state.lm,
                                        1,
                                        width,
                                        x,
                                        y,
                                    )
                                    density_time.append(dtime[0, 0])

                    elif choose_d_method == "Classical":
                        density_ret = np.zeros((len(ybins) - 1, len(xbins) - 1))
                        density_ret = Utilities.calculate_density_average_classic(
                            geominX,
                            geomaxX,
                            geominY,
                            geomaxY,
                            dx,
                            len(frames),
                            data[:, 2],
                            data[:, 3],
                        )
                        # time serie
                        density_time = []
                        if choose_timeseries:
                            for frame in frames[::sample]:
                                dframe = data[:, 1] == frame
                                x = data[dframe][:, 2]
                                y = data[dframe][:, 3]
                                dtime = Utilities.calculate_density_frame_classic(
                                    st.session_state.xpos - st.session_state.lm / 2,
                                    st.session_state.xpos + st.session_state.lm / 2,
                                    st.session_state.ypos - st.session_state.lm / 2,
                                    st.session_state.ypos + st.session_state.lm / 2,
                                    st.session_state.lm,
                                    x,
                                    y,
                                )
                                density_time.append(dtime[0, 0])
                    st.session_state.density = density_ret
                    msg += f"Density profile in range [{np.min(density_ret):.2f} : {np.max(density_ret):.2f}] [1/m^2]. \n"
                    fig = plots.plot_profile_and_geometry2(
                        xbins,
                        ybins,
                        geometry_wall,
                        st.session_state.xpos,
                        st.session_state.ypos,
                        st.session_state.lm,
                        density_ret,
                        interpolation,
                        label=r"1/m/m",
                        title="Density",
                        vmin=None,
                        vmax=None,
                    )
                    dprofile_pl.plotly_chart(fig, use_container_width=True)
                    if choose_timeseries:
                        fig = plots.plot_timeserie(
                            frames,
                            density_time,
                            fps,
                            "Density / m / m",
                            np.min(density_time),
                            np.max(density_time) + 2,
                            np.max(density_ret)
                        )
                        dtimeseries_pl.plotly_chart(fig, use_container_width=True)

                if choose_vprofile:
                    if choose_d_method == "Gaussian":
                        speed_ret = Utilities.weidmann(st.session_state.density)
                        if choose_timeseries:
                            speed_time = Utilities.weidmann(np.array(density_time))

                    else:
                        speed_ret = Utilities.calculate_speed_average(
                            geominX,
                            geomaxX,
                            geominY,
                            geomaxY,
                            dx,
                            len(frames),
                            data[:, 2],
                            data[:, 3],
                            data[:, st.session_state.speed_index],
                        )
                        if choose_timeseries:
                            speed_time = []
                            for frame in frames[::sample]:
                                dframe = data[:, 1] == frame
                                x = data[dframe][:, 2]
                                y = data[dframe][:, 3]
                                speed_agent = data[dframe][
                                    :, st.session_state.speed_index
                                ]
                                stime = Utilities.calculate_speed_average(
                                    st.session_state.xpos - st.session_state.lm / 2,
                                    st.session_state.xpos + st.session_state.lm / 2,
                                    st.session_state.ypos - st.session_state.lm / 2,
                                    st.session_state.ypos + st.session_state.lm / 2,
                                    st.session_state.lm,
                                    1,
                                    x,
                                    y,
                                    speed_agent,
                                )
                                speed_time.append(stime[0, 0])

                    fig = plots.plot_profile_and_geometry2(
                        xbins,
                        ybins,
                        geometry_wall,
                        st.session_state.xpos,
                        st.session_state.ypos,
                        st.session_state.lm,
                        speed_ret,
                        interpolation,
                        label=r"v / m/s",
                        title="Speed",
                        vmin=None,
                        vmax=None,
                    )
                    vprofile_pl.plotly_chart(fig, use_container_width=True)
                    if choose_timeseries:
                        fig = plots.plot_timeserie(
                            frames,
                            speed_time,
                            fps,
                            "Speed / m/s",
                            np.min(speed_time),
                            np.max(speed_time),
                            np.max(speed_ret),
                        )
                        vtimeseries_pl.plotly_chart(fig, use_container_width=True)

                    speed = data[:, st.session_state.speed_index]
                    msg += f"Speed profile in range [{np.min(speed_ret):.2f} : {np.max(speed_ret):.2f}] [m/s]. "
                    msg += f"Speed trajectory in range [{np.min(speed):.2f} : {np.max(speed):.2f}] [m/s]. "

                    messages.info(msg)

            # RSET
            rset_max = Utilities.calculate_RSET(geominX, geomaxX, geominY, geomaxY, dx,
                                                data[:, 2],
                                                data[:, 3],
                                                data[:, 1]/fps,
                                                "max")
            # rset_min = Utilities.calculate_RSET(geominX, geomaxX, geominY, geomaxY, dx,
            #                                     data[:, 2],
            #                                     data[:, 3],
            #                                     data[:, 1]/fps,
            #                                     "min")
            # fig = plots.plot_profile_and_geometry2(
            #     xbins,
            #     ybins,
            #     geometry_wall,
            #     None, None, None,
            #     rset_min,
            #     interpolation,
            #     label=r"first arrival time / s",
            #     title="RSET min",
            #     vmin=None,
            #     vmax=None,
            # )
            # rset_pl.plotly_chart(fig, use_container_width=True)
            nbins = 10
            fig = plots.plot_RSET_hist(rset_max, nbins)
            rset_pl.plotly_chart(fig, use_container_width=True)
            
            fig = plots.plot_profile_and_geometry2(
                xbins,
                ybins,
                geometry_wall,
                None, None, None,
                rset_max,
                interpolation,
                label=r"time / s",
                title=f"RSET = {np.max(rset_max):.1f} / s",
                vmin=None,
                vmax=None,
            )
            rset2_pl.plotly_chart(fig, use_container_width=True)
            
        # todo
        info = st.expander("Documentation: Plot curves (click to expand)")
        with info:
            doc.doc_plots()

        plot_options = choose_NT or choose_flow or choose_time_distance or choose_survival
        # all these options need to calculate N-T-Data
        if make_plots and plot_options:
            # todo: cache calculation in st.session_state.tstats
            with Utilities.profile("calculate_NT_data"):
                tstats, cum_num, cum_num_positiv, cum_num_negativ, trans_used, max_len, msg = Utilities.calculate_NT_data(
                    transitions,
                    selected_transitions,
                    data,
                    fps,
                )

        if make_plots:
            c1, c2, c3 = st.columns((1, 1, 1))
            if choose_NT:
                peds_inside = Utilities.peds_inside(data)
                fig1 = plots.plot_peds_inside(frames, peds_inside, fps)
                c2.plotly_chart(fig1, use_container_width=True)
                if tstats:
                    fig2 = plots.plot_NT(tstats, cum_num, cum_num_positiv, cum_num_negativ, fps)
                    c1.plotly_chart(fig2, use_container_width=True)                                
            
                if choose_flow and tstats:
                    fig = plots.plot_flow(tstats, cum_num, cum_num_positiv, cum_num_negativ, fps)
                    c3.plotly_chart(fig, use_container_width=True)
                    
        c1, c2 = st.columns((1, 1))
        if make_plots and choose_time_distance:
            with c1:
                with Utilities.profile("plot distance-time curve"):
                    selected_and_used_transitions = [
                        i for i in selected_transitions if trans_used[i]
                    ]
                    if selected_and_used_transitions:
                        i = selected_and_used_transitions[0]
                        Frames = tstats[i]
                        fig = plots.plot_time_distance(
                            Frames,
                            data,
                            LineString(transitions[i]),
                            i,
                            fps,
                            num_peds_TD,
                            sample_TD,
                        )
                        st.plotly_chart(fig, use_container_width=True)

        if make_plots and choose_survival:
            with c2:
                if selected_transitions:
                    fig = plots.plot_survival(tstats, fps)
                    st.plotly_chart(fig, use_container_width=True)

        if make_plots:
            if selected_transitions:
                st.info(msg)

        # -- download stats
        if make_plots:
            if choose_NT:
                T = dt.datetime.now()
                if traj_from_upload:
                    n = trajectory_file.name.split(".txt")[0]
                else:
                    n = trajectory_file_d.split(".txt")[0]

                file_download = f"{n}_{T.year}-{T.month:02}-{T.day:02}_{T.hour:02}-{T.minute:02}-{T.second:02}.txt"
                once = 0  # don't download if file is empty
                for i in selected_transitions:
                    if not trans_used[i]:
                        continue

                    nrows = tstats[i].shape[0]
                    if nrows < max_len:
                        tmp_stats = np.full((max_len, 3), -1)
                        tmp_stats[: nrows, :] = tstats[i]
                        tmp_cum_num = np.full(max_len, -1)
                        tmp_cum_num[: len(cum_num[i])] = cum_num[i]
                    else:
                        tmp_stats = tstats[i]
                        tmp_cum_num = cum_num[i]
                        
                    tmp_cum_num_p = np.full(len(tmp_cum_num), -1)
                    tmp_cum_num_p[: len(cum_num_positiv[i])] = cum_num_positiv[i]
                    tmp_cum_num_n = np.full(len(tmp_cum_num), -1)
                    tmp_cum_num_n[: len(cum_num_negativ[i])] = cum_num_negativ[i]

                    tmp_cum_num = tmp_cum_num.reshape(len(tmp_cum_num), 1)
                    tmp_cum_num_p = tmp_cum_num_p.reshape(len(tmp_cum_num_p), 1)
                    tmp_cum_num_n = tmp_cum_num_n.reshape(len(tmp_cum_num_n), 1)
                    if not once:
                        all_stats = np.hstack((tmp_stats, tmp_cum_num, tmp_cum_num_p, tmp_cum_num_n))
                        once = 1
                    else:
                        all_stats = np.hstack((all_stats, tmp_stats, tmp_cum_num, tmp_cum_num_p, tmp_cum_num_n))

                if selected_transitions and once:
                    passed_lines = [i for i in selected_transitions if trans_used[i]]
                    fmt = len(passed_lines) * ["%d", "%d", "%d", "%d", "%d", "%d"]
                    # all_stats = all_stats.T
                    np.savetxt(
                        file_download,
                        all_stats,
                        fmt=fmt,
                        header="line ids: \n"
                        + np.array2string(
                            np.array(passed_lines, dtype=int),
                            precision=2,
                            separator="\t",
                            suppress_small=True,
                        )
                        + f"\nframerate: {fps:.0f}"
                        + "\npid\tframe\tdirection\tcount_tot\tcount+\tcount-",
                        comments="#",
                        delimiter="\t",
                    )
                    with open(file_download, encoding="utf-8") as f:
                        pc.download_button(
                            "Download statistics", f, file_name=file_download
                        )

        info = st.expander("Documentation: Jam (click to expand)")
        with info:
            doc.doc_jam()

        if choose_jam_duration:
            logging.info("calculate jam")
            logging.info(f"jam speed {jam_speed}")
            logging.info(f"min jam agents {min_jam_agents}")
            logging.info(f"min jam time {min_jam_time}")
        
            c1, c2 = st.columns((1, 1))
            pl2 = c1.empty()
            pl = c2.empty()

            precision = c1.slider(
                "Precision",
                0,
                int(10 * fps),
                help="Condition on the length of jam durations (in frame)",
            )
            nbins = c2.slider(
                "Number of bins", 5, 40, value=10, help="Number of bins", key="lifetime"
            )

            pl3 = c1.empty()
            pl4 = c1.empty()
            nbins2 = pl4.slider(
                "Number of bins", 5, 40, value=10, help="Number of bins", key="waiting"
            )

            ##  lifetime
            jam_frames = Utilities.jam_frames(data, jam_speed)
            with Utilities.profile("jam_lifetime"):
                lifetime, chuncks, max_lifetime, from_to = Utilities.jam_lifetime(
                    data, jam_frames[10:], min_jam_agents, fps, precision
                )  # remove the first frames, cause in simulation people stand

            ## duration
            logging.info(f"waiting time with {min_jam_time}")
            with Utilities.profile("jam_waiting_time"):
                waiting_time = Utilities.jam_waiting_time(
                    data, jam_speed, min_jam_time, fps, precision
                )

            if not waiting_time.size:
                wtimes = np.array([])
            else:
                wtimes = waiting_time[:, 1]

            with Utilities.profile("Rendering Jam figures"):
                ## plots
                fig1 = plots.plot_jam_lifetime(
                    frames, lifetime, fps, max_lifetime, from_to, min_jam_agents
                )
                hist = plots.plot_jam_lifetime_hist(chuncks, fps, nbins)
                pl2.plotly_chart(fig1, use_container_width=True)
                pl.plotly_chart(hist, use_container_width=True)
                # --
                hist = plots.plot_jam_waiting_hist(wtimes, fps, nbins2)
                pl3.plotly_chart(hist, use_container_width=True)

    time_end = timeit.default_timer()
    msg_time = Utilities.get_time(time_end - time_start)
    time_msg.info(f":clock8: Finished in {msg_time}")


if __name__ == "__main__":
    with Utilities.profile("Main"):
        main()
