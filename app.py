import datetime as dt
import os
from collections import defaultdict
from io import StringIO
from pathlib import Path
from xml.dom.minidom import parseString

import lovely_logger as logging
import numpy as np
import streamlit as st
from matplotlib import cm
from shapely.geometry import LineString

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
        st.session_state.data = []

    if "frames" not in st.session_state:
        st.session_state.frames = []

    if "fps" not in st.session_state:
        st.session_state.fps = 16

    if "peds" not in st.session_state:
        st.session_state.peds = []

    if "conv" not in st.session_state:
        st.session_state.conv = False

    if "density" not in st.session_state:
        st.session_state.density = []

    if "tstats" not in st.session_state:
        st.session_state.tstats = defaultdict(list)

    if "cum_num" not in st.session_state:
        st.session_state.cum_num = {}

    if "speed_index" not in st.session_state:
        st.session_state.speed_index = 9


if __name__ == "__main__":
    st.header(":information_source: Dashboard")
    info = st.expander("click to expand")
    with info:
        Utilities.docs()

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
    # st.sidebar.markdown("-------")
    geometry_file = c2.file_uploader(
        "🏠 Geometry file ",
        type=["xml"],
        help="Load geometry file",
    )
    st.sidebar.markdown("-------")
    unit = st.sidebar.radio(
        "Trajectories are in ",
        ["m", "cm"],
        help="Choose the unit of the original trajectories. Data in the app will be converted to meter",
    )
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("-------")
    st.sidebar.header("Plot")
    c1, c2 = st.sidebar.columns((1, 1))
    choose_trajectories = c1.checkbox(
        "Trajectories", help="Plot trajectories", key="Traj"
    )
    if choose_trajectories:
        choose_transitions = c2.checkbox(
            "Transitions", help="Show transittions", key="Tran"
        )

    pl = st.sidebar.empty()

    st.sidebar.markdown("-------")
    st.sidebar.header("Speed")
    how_speed = st.sidebar.radio("Source:", ["from trajectory", "from simulation"])
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    if how_speed == "from trajectory":
        df = st.sidebar.slider(
            "df",
            2,
            50,
            10,
            help="how many frames to consider for calculating the speed",
        )

    st.sidebar.markdown("-------")
    st.sidebar.header("Profile")
    c1, c2 = st.sidebar.columns((1, 1))
    choose_dprofile = c1.checkbox(
        "Show", help="Plot density and speed profiles", key="dProfile"
    )
    # choose_vprofile = c2.checkbox("Speed", help="Plot speed profile", key="vProfile")
    choose_d_method = st.sidebar.radio(
        "Density method",
        ["Classical", "Gaussian", "Weidmann"],
        help="""
        How to calculate average of density over time and space""",
    )
    # choose_v_method = st.sidebar.radio(
    #     "Speed method",
    #     ["Speed", "Density"],
    #     help="""
    #                                  How to calculate average of speed over time and space""",
    # )

    # st.write(
    #     "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
    #     unsafe_allow_html=True,
    # )
    if choose_d_method == "Gaussian":
        width = st.sidebar.slider(
            "Width", 0.05, 1.0, 0.3, help="Width of Gaussian function"
        )

    dx = st.sidebar.slider("Step", 0.01, 1.0, 0.5, help="Space discretization")
    methods = ["nearest", "gaussian", "sinc", "bicubic", "mitchell", "bilinear"]
    interpolation = st.sidebar.radio(
        "Interpolation", methods, help="Interpolation method for imshow()"
    )
    st.sidebar.markdown("-------")
    st.sidebar.header("Plot curves")
    c1, c2 = st.sidebar.columns((1, 1))
    msg_status = st.sidebar.empty()
    disable_NT_flow = False
    if trajectory_file and geometry_file:
        logging.info(f">> {trajectory_file.name}")
        logging.info(f">> {geometry_file.name}")
        try:

            h = st.expander("Trajectories (first 4 columns)", expanded=True)
            stringio = StringIO(trajectory_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            new_data = False
            if string_data != st.session_state.old_data:
                st.session_state.old_data = string_data
                new_data = True
                logging.info("Loaded new trajectories")

            if new_data:
                logging.info("Load trajectories")
                data = Utilities.read_trajectory(trajectory_file)
                fps = Utilities.get_fps(string_data)
                peds = np.unique(data[:, 0]).astype(int)
                frames = np.unique(data[:, 1])

                st.session_state.data = data
                st.session_state.fps = fps
                st.session_state.peds = peds
                st.session_state.frames = frames
                st.session_state.conv = False

                with h:
                    plots.show_trajectories_table(data)

            else:
                data = st.session_state.data
                fps = st.session_state.fps
                peds = st.session_state.peds
                frames = st.session_state.frames

            if st.session_state.conv is False and unit == "cm":
                data[:, 2:] /= 100
                st.session_state.conv = True

            if st.session_state.conv is True and unit == "m":
                data[:, 2:] *= 100
                st.session_state.conv = False

            st.markdown("### :bar_chart: Statistics")
            pl_msg = st.empty()
            msg = f"""
            Frames per second: {fps}\n
            Frames: {len(frames)} | First: {frames[0]:.0f} | Last: {frames[-1]:.0f}\n
            Agents: {len(peds)}\n
            Evac-time: {np.max(data[:, 1])/fps} [s]
            """
            pl_msg.info(msg)
            plot_ped = pl.select_slider(
                "Highlight pedestrian", options=peds, value=(peds[10])
            )

            logging.info(f"fps = {fps}")
        except Exception as e:
            msg_status.error(
                f"""Can't parse trajectory file.
                Error: {e}"""
            )
            st.stop()

        try:
            parse_geometry_file = Utilities.get_geometry_file(string_data)
            logging.info(f"Geometry: <{geometry_file.name}>")
            # Read Geometry file
            if parse_geometry_file != geometry_file.name:
                st.error(
                    f"Mismatched geometry files. Parsed {parse_geometry_file}. Uploaded {geometry_file.name}"
                )
                st.stop()

            stringio = StringIO(geometry_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            file_data = geometry_file.read()

            geo_xml = parseString(geometry_file.getvalue())
            logging.info("Geometry parsed successfully")
            geometry_wall = Utilities.read_subroom_walls(geo_xml, unit)
            logging.info("Got geometry walls successfully")
            transitions = Utilities.get_transitions(geo_xml, unit)
            logging.info("Got geometry transitions successfully")
            measurement_lines = Utilities.get_measurement_lines(geo_xml, unit)
            logging.info("Got geometry measurement_lines successfully")
            # todo: check if ids of transitions and measurement_lines are unique
            transitions.update(measurement_lines)
            # select all per default
            if transitions:
                default = list(transitions.keys())  # choose this transition by default
            else:
                default = []
                disable_NT_flow = True

            logging.info("Get geo_limits")
            geominX, geomaxX, geominY, geomaxY = Utilities.geo_limits(geo_xml, unit)

            logging.info(
                f"GeometrySize: X: ({geominX:.2f},{geomaxX:.2f}), Y: ({geominY:.2f},{geominY:.2f})"
            )
        except Exception as e:
            msg_status.error(
                f"""Can't parse geometry file.
                Error: {e}"""
            )
            st.stop()

        NT_form = st.sidebar.form("plot-NT")
        with NT_form:
            choose_NT = c1.checkbox(
                "N-T", help="Plot N-t curve", key="NT", disabled=disable_NT_flow
            )

            choose_flow = c2.checkbox(
                "Flow", help="Plot flow curve", key="Flow", disabled=disable_NT_flow
            )
            choose_evactime = c1.checkbox(
                "Occupation",
                help="Plot number of pedestrians inside geometry over time",
                key="EvacT",
            )

        selected_transitions = NT_form.multiselect(
            "Select transition",
            transitions.keys(),
            default,
            help="Transition to calculate N-T. Can select multiple transitions",
        )
        make_plots = NT_form.form_submit_button(label="🚦plot")
        # ----- Jam
        st.sidebar.header("Jam")
        st.sidebar.slider(
            "Min jam speed",
            0.1,
            1.0,
            0.5,
            help="Agent slower that this speed is in jam",
            key="jVmin",
        )
        st.sidebar.slider(
            "Min jam duration",
            1,
            300,
            60,
            help="A jam last at least that long",
            key="jTmin",
        )
        st.sidebar.slider(
            "Min agents in jam",
            2,
            50,
            20,
            help="A jam has at least so many agents",
            key="jNmin",
        )

        if disable_NT_flow:
            st.sidebar.info("N-T and Flow plots are disabled, because no transitions!")

        if how_speed == "from simulation":
            logging.info("speed by simulation")
            Utilities.check_shape_and_stop(data.shape[1], how_speed)
            speed = data[:, 9]
            st.session_state.speed_index = 9
        else:
            logging.info("speed by trajectory")
            speed = Utilities.compute_speed(data, fps, df)
            st.session_state.speed_index = -1

        if choose_trajectories:
            logging.info("plotting trajectories")
            agent = data[data[:, 0] == plot_ped]
            # print("index", st.session_state.speed_index)
            # speed_agent1 = agent[:, st.session_state.speed_index]
            if how_speed == "from simulation":
                speed_agent = agent[:, 9]
                angle_agent = agent[:, 7]
            else:
                speed_agent, angle_agent = Utilities.compute_agent_speed_and_angle(agent, fps, df)

            c1, c2 = st.columns((1, 1))
            logging.info(f"Pedesrians: [{plot_ped}]")
            with c1:
                if choose_transitions:
                    plots.plot_trajectories(
                        data,
                        plot_ped,
                        speed_agent,
                        geometry_wall,
                        transitions,
                        geominX,
                        geomaxX,
                        geominY,
                        geomaxY,
                    )
                else:
                    plots.plot_trajectories(
                        data,
                        plot_ped,
                        speed_agent,
                        geometry_wall,
                        {},
                        geominX,
                        geomaxX,
                        geominY,
                        geomaxY,
                    )

            with c2:
                plots.plot_agent_xy(agent[:, 1], agent[:, 2], agent[:, 3], fps)

            with c1:
                plots.plot_agent_angle(plot_ped, agent[:, 1], angle_agent, fps)

            with c2:
                plots.plot_agent_speed(
                    plot_ped, agent[:, 1], speed_agent, np.max(speed), fps
                )

        # choose_dprofile =
        choose_vprofile = True  # todo: not sure is I want to keep this option
        if choose_dprofile or choose_vprofile:
            Utilities.check_shape_and_stop(data.shape[1], how_speed)
            msg = ""
            with st.spinner("Processing ..."):
                c1, _, c2 = st.columns((1, 0.05, 1))
                if choose_dprofile:
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
                            speed,
                        )
                    elif choose_d_method == "Gaussian":
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
                    elif choose_d_method == "Classical":
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

                    st.session_state.density = density_ret
                    msg += f"Density in range [{np.min(density_ret):.2f} : {np.max(density_ret):.2f}] [1/m^2]. "
                    with c1:
                        plots.plot_profile_and_geometry(
                            geominX,
                            geomaxX,
                            geominY,
                            geomaxY,
                            geometry_wall,
                            density_ret,
                            interpolation,
                            cmap=cm.jet,
                            label=r"$\rho\; / 1/m^2$",
                            title="Density",
                            vmin=None,
                            vmax=None,
                        )
                    if choose_vprofile:
                        if choose_d_method == "Gaussian":
                            speed_ret = speed = Utilities.weidmann(
                                st.session_state.density
                            )
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
                                speed,
                            )
                            print(speed, len(frames))
                        with c2:
                            plots.plot_profile_and_geometry(
                                geominX,
                                geomaxX,
                                geominY,
                                geomaxY,
                                geometry_wall,
                                speed_ret,
                                interpolation,
                                cmap=cm.jet.reversed(),
                                label=r"$v\; / m/s$",
                                title="Speed",
                                vmin=None,
                                vmax=None,
                            )
                        msg += f"Speed profile in range [{np.min(speed_ret):.2f} : {np.max(speed_ret):.2f}] [m/s]. "
                        msg += f"Speed trajectory in range [{np.min(speed):.2f} : {np.max(speed):.2f}] [m/s]. "

                    st.info(msg)

        # todo
        choose_speed = True
        plot_options = choose_NT or choose_flow or choose_speed
        if make_plots and plot_options:
            peds = np.unique(data)
            tstats, cum_num, trans_used, max_len = Utilities.calculate_NT_data(
                transitions, selected_transitions, data, fps
            )
            st.session_state.tstats = tstats
            st.session_state.cum_num = cum_num

        if make_plots:
            c1, c2 = st.columns((1, 1))
            with c1:
                if choose_NT and tstats:
                    plots.plot_NT(tstats, cum_num, fps)

            with c2:
                if choose_flow and tstats:
                    plots.plot_flow(tstats, cum_num, fps)

        c1, c2 = st.columns((1, 1))
        if make_plots and choose_evactime:
            peds_inside = []
            for frame in frames:
                d = data[data[:, 1] == frame][:, 0]
                peds_inside.append(len(d))

            with c1:
                plots.plot_peds_inside(frames, peds_inside, fps)

            with c2:
                # plots.plot_speed(tstats, fps)
                pass

        # -- download stats
        if make_plots:
            if choose_NT:
                T = dt.datetime.now()
                n = trajectory_file.name.split(".txt")[0]
                file_download = f"{n}_{T.year}-{T.month:02}-{T.day:02}_{T.hour:02}-{T.minute:02}-{T.second:02}.txt"
                once = 0  # don't download if file is empty
                for i in selected_transitions:
                    if not trans_used[i]:
                        continue

                    if len(tstats[i]) < max_len:
                        tmp_stats = np.full(max_len, -1)
                        tmp_stats[: len(tstats[i])] = tstats[i]
                        tmp_cum_num = np.full(max_len, -1)
                        tmp_cum_num[: len(cum_num[i])] = cum_num[i]
                    else:
                        tmp_stats = tstats[i]
                        tmp_cum_num = cum_num[i]

                    if not once:
                        all_stats = np.vstack((tmp_stats, tmp_cum_num))
                        once = 1
                    else:
                        all_stats = np.vstack((all_stats, tmp_stats, tmp_cum_num))

                if selected_transitions and once:
                    passed_lines = [i for i in selected_transitions if trans_used[i]]
                    fmt = len(passed_lines) * ["%d", "%d"]
                    all_stats = all_stats.T
                    np.savetxt(
                        file_download,
                        all_stats,
                        fmt=fmt,
                        header=np.array2string(
                            np.array(passed_lines, dtype=int),
                            precision=2,
                            separator="\t",
                            suppress_small=True,
                        ),
                        comments="#",
                        delimiter="\t",
                    )
                    with open(file_download, encoding="utf-8") as f:
                        download = st.sidebar.download_button(
                            "Download statistics", f, file_name=file_download
                        )
