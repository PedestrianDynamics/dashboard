import datetime as dt
import os
from collections import defaultdict
from io import StringIO
from pathlib import Path
from xml.dom.minidom import parseString

import lovely_logger as logging
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# import matplotlib.cm as cm
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import read_csv
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


def set_state_variables():
    if "old_configs" not in st.session_state:
        st.session_state.old_configs = ""


def read_trajectory(input_file):
    data = read_csv(input_file, sep=r"\s+", dtype=np.float64, comment="#").values

    return data


if __name__ == "__main__":
    st.header(":information_source: Dashboard")
    info = st.expander("click to expand")
    with info:
        st.write(
            """
         This app performs some basic measurements on data simulated by jpscore.

         #### Speed
         The speed can be calculated *from simulation*: in this case
         use in the inifile the option: `<optional_output   speed=\"TRUE\">`.

         Alternatively, the speed can be calculated *from trajectory*
         according to the forward-formula:
         """
        )
        st.latex(
            r"""
         \begin{equation}
         v_i(f) = \frac{x_i(f+df) - x_i(f))}{df},
         \end{equation}
         """
        )
        st.write(
            r"""with $df$ a constant and $v_i(f)$ the speed of pedestrian $i$ at frame $f$."""
        )
        st.write(
            """
         #### Density
         The density is calculated based on the speed (1) using the Weidmann-formula **[Weidmann1992 Eq. (15)]**:
         """
        )
        st.latex(
            r"""
         \begin{equation}
         v_i = v^0 \Big(1 - \exp\big(\gamma (\frac{1}{\rho_i} - \frac{1}{\rho_{\max}}) \big)  \Big).
         \end{equation}
         """
        )
        st.text("Eq. (2) can be transformed in ")
        st.latex(
            r"""
         \begin{equation*}
         \rho_i = \Big(-\frac{1}{\gamma} \log(1 - \frac{v_i}{v^0})+ \frac{1}{\rho_{\max}}\Big)^{-1},
         \end{equation*}
         """
        )
        st.write("""where""")
        st.latex(
            r"""\gamma = 1.913\, m^{-2},\; \rho_{\max} = 5.4\, m^{-2}\; \;{\rm and}\; v^0 = 1.34\, m/s."""
        )
        st.markdown("--------")
        st.write("#### References:")
        st.code(
            "Weidmann1992: U. Weidmann, Transporttechnik der Fussgänger: Transporttechnische Eigenschaften des Fussgängerverkehrs, Literaturauswertung, 1992"
        )
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
    unit = st.sidebar.radio("Unit of trajectories", ["m", "cm"])
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
    choose_transitions = c2.checkbox(
        "Transitions", help="Show transittions", key="Tran"
    )
    st.sidebar.markdown("-------")
    st.sidebar.header("Profile")
    c1, c2 = st.sidebar.columns((1, 1))
    choose_dprofile = c1.checkbox(
        "Density", help="Plot density profile", key="dProfile"
    )
    choose_vprofile = c2.checkbox(
        "Speed", help="Plot speed profile", key="vProfile"
    )
    how_speed = st.sidebar.radio("Speed", ["from simulation", "from trajectory"])
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

    dx = st.sidebar.slider("Step", 0.01, 1.0, 0.5, help="Space discretization")
    methods = ["nearest", "gaussian", "sinc", "bicubic", "mitchell", "bilinear"]
    interpolation = st.sidebar.radio(
        "Method", methods, help="Interpolation methods for imshow()"
    )
    st.sidebar.markdown("-------")
    st.sidebar.header("Plot curves")
    c1, c2 = st.sidebar.columns((1, 1))
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
        unsafe_allow_html=True,
    )
    msg_status = st.sidebar.empty()
    disable_NT_flow = False
    if trajectory_file and geometry_file:
        logging.info(f">> {trajectory_file.name}")
        logging.info(f">> {geometry_file.name}")
        try:
            data = read_trajectory(trajectory_file)
            if unit == "cm":
                data[:, 2:] /= 100

            h = st.expander("Head of trajectory")
            with h:
                st.markdown("### Head of trajectories")
                st.table(data[:10, :])  # will display the table
            stringio = StringIO(trajectory_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            fps = Utilities.get_fps(string_data)
            peds = np.unique(data[:, 0])
            st.markdown("### :bar_chart: Statistics")
            pl = st.empty()
            msg = f"""
            Frames per second: {fps}\n
            Agents: {len(peds)}\n
            Evac-time: {np.max(data[:, 1])/fps} [s]
            """
            pl.info(msg)
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
            logging.info(transitions)
            if transitions:
                default = list(transitions.keys())[0]
            else:
                default = []
                disable_NT_flow = True

            selected_transitions = st.sidebar.multiselect(
                "Select transition",
                transitions.keys(),
                default,
                help="Transition to calculate N-T. Can select multiple transitions",
            )
            logging.info("get geo_limits")
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

        choose_NT = c1.checkbox("N-T", help="Plot N-t curve", key="NT", disabled=disable_NT_flow)
        choose_flow = c2.checkbox("Flow", help="Plot flow curve", key="Flow", disabled=disable_NT_flow)
        if disable_NT_flow:
            st.sidebar.info("N-T and Flow plots are disabled, because no transitions!")
        if choose_trajectories:
            logging.info("plotting trajectories")
            if choose_transitions:
                Utilities.plot_trajectories(
                    data, geometry_wall, transitions, geominX, geomaxX, geominY, geomaxY
                )
            else:
                Utilities.plot_trajectories(
                    data, geometry_wall, {}, geominX, geomaxX, geominY, geomaxY
                )

        if how_speed == "from simulation":
            logging.info("speed by simulation")
            Utilities.check_shape_and_stop(data.shape[1], how_speed)
            speed = data[:, 9]
        else:
            logging.info("speed by trajectory")
            speed = Utilities.compute_speed(data, fps, df)

        if choose_dprofile or choose_vprofile:
            Utilities.check_shape_and_stop(data.shape[1], how_speed)
            c1, _, c2 = st.columns((1, 0.05, 1))
            msg = ""
            with c1:
                if choose_dprofile:
                    density = Utilities.weidmann(speed)
                    Utilities.plot_profile(
                        geominX,
                        geomaxX,
                        geominY,
                        geomaxY,
                        geometry_wall,
                        dx,
                        data[:, 2],
                        data[:, 3],
                        density,
                        vmin=0,
                        vmax=6,
                        interpolation=interpolation,
                        cmap=cm.jet,
                        label=r"$\rho\; / 1/m^2$",
                        title="Density",
                    )
                    msg  += f"Density in range [{np.min(density):.2f} : {np.max(density):.2f}] [1/m^2]. "
            with c2:
                if choose_vprofile:
                    Utilities.plot_profile(
                        geominX,
                        geomaxX,
                        geominY,
                        geomaxY,
                        geometry_wall,
                        dx,
                        data[:, 2],
                        data[:, 3],
                        speed,
                        vmin=0,
                        vmax=1.3,
                        interpolation=interpolation,
                        cmap=cm.jet.reversed(),
                        label=r"$v\; / m/s$",
                        title="Speed",
                    )
                    msg += f"Speed in range [{np.min(speed):.2f} : {np.max(speed):.2f}] [m/s]. "
            st.info(msg)

        if choose_NT or choose_flow:
            peds = np.unique(data)
            tstats = defaultdict(list)
            cum_num = {}
            msg = ""
            trans_used = {}
            with st.spinner('Processing ...'):
                max_len = -1  # longest array. Needed to stack arrays and save them in file
                for i, t in transitions.items():
                    trans_used[i] = False
                    if i in selected_transitions:
                        line = LineString(t)
                        for ped in peds:
                            ped_data = data[data[:, 0] == ped]
                            frame = Utilities.passing_frame(ped_data, line, fps)
                            if frame >= 0:
                                tstats[i].append(frame)
                                trans_used[i] = True

                        if trans_used[i]:
                            tstats[i].sort()
                            cum_num[i] = np.cumsum(np.ones(len(tstats[i])))
                            flow = cum_num[i][-1] / tstats[i][-1] * fps                            
                            max_len = max(max_len, cum_num[i].size)
                            msg += f"Transition {i},  flow: {flow:.2f} [1/s] \n \n"
                        else:
                            msg += f"Transition {i},  flow: 0 [1/s] \n \n"

        c1, _, c2 = st.columns((1, 0.05, 1))

        with c1:
            if choose_NT and tstats:
                Utilities.plot_NT(tstats, cum_num, fps)

        with c2:
            if choose_flow and tstats:
                Utilities.plot_flow(tstats, cum_num, fps)

            # -- download stats
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
                    tmp_stats[:len(tstats[i])] = tstats[i]
                    tmp_cum_num = np.full(max_len, -1)
                    tmp_cum_num[:len(cum_num[i])] = cum_num[i]
                else:
                    tmp_stats = tstats[i]
                    tmp_cum_num = cum_num[i]

                if not once:
                    all_stats = np.vstack((tmp_stats, tmp_cum_num))
                    once = 1
                else:
                    all_stats = np.vstack((all_stats, tmp_stats, tmp_cum_num))
                    
            if selected_transitions:
                st.info(msg)

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
