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
         st.write("""
         This app performs some basic measurements on data simulated by jpscore.

         #### Speed
         The speed can be calculated *from simulation*: in this case
         use in the inifile the option: `<optional_output   speed=\"TRUE\">`.

         Alternatively, the speed can be calculated *from trajectory*
         according to the forward-formula:
         """)
         st.latex(r'''
         \begin{equation}
         v_i(f) = \frac{x_i(f+df) - x_i(f))}{df},
         \end{equation}
         ''')
         st.text("""with""")
         st.latex(r"""df\; {\rm a\; constant\; and}\; v_i(f)\; {\rm the\; speed\; of\; pedestrian}\; i\; {\rm at}\; f.""")
         st.write("""
         #### Density
         The density is calculated based on the speed (1) using the Weidmann-formula **[Weidmann1992 Eq. (15)]**:
         """)
         st.latex(r'''
         \begin{equation*}
         \frac{1}{\rho} = \frac{-1}{\gamma} \log(1 - \frac{v_i}{v^0})+ \frac{1}{\rho_{\max}},
         \end{equation*}
         ''')
         st.write("""where""")
         st.latex(r"""\gamma = 1.913\, m^{-2},\; \rho_{\max} = 5.4\, m^{-2}\; \;{\rm and}\; v^0 = 1.34 m/s.""")
         st.markdown("--------")
         st.write("#### References:")
         st.code("Weidmann1992: U. Weidmann, Transporttechnik der Fussgänger: Transporttechnische Eigenschaften des Fussgängerverkehrs, Literaturauswertung, 1992")
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
        "Velocity", help="Plot velocity profile", key="vProfile"
    )
    how_speed = st.sidebar.radio("Speed", ['from simulation', 'from trajectory'])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    if how_speed == "from trajectory":
        df = st.sidebar.slider("df", 2, 50, 10, help="how many frames to consider for calculating the speed")

    dx = st.sidebar.slider("Step", 0.01, 1.0, 0.5, help="Space discretization")
    methods = ['nearest', 'bilinear', 'sinc']
    interpolation = st.sidebar.radio("Method", methods)
    st.sidebar.markdown("-------")
    st.sidebar.header("N-T diagram")
    choose_NT = st.sidebar.checkbox("Plot", help="Plot N-t curve", key="NT")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    msg_status = st.sidebar.empty()
    
    if trajectory_file and geometry_file:
        logging.info(f">> {trajectory_file.name}")
        logging.info(f">> {geometry_file.name}")
        try:
            data = read_trajectory(trajectory_file)
            st.markdown("### Head of trajectories")
            st.table(data[:10, :])# will display the table
            stringio = StringIO(trajectory_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            fps = Utilities.get_fps(string_data)
            peds = np.unique(data[:, 0])
            st.markdown("### :bar_chart: Statistics")
            pl = st.empty()
            msg = f"""
            Frames per second: {fps}\n
            Agents: {len(peds)}\n
            Evac-time: {np.max(data[:, 1])/fps} seconds
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
            geometry_wall = Utilities.read_subroom_walls(geo_xml)
            logging.info("Got geometry walls successfully")
            transitions = Utilities.get_transitions(geo_xml)
            logging.info("Got geometry transitions successfully")
            selected_transitions = st.sidebar.multiselect(
                "Select transition",
                transitions.keys(),
                list(transitions.keys())[0],
                help="Transition to calculate N-T. Can select multiple transitions",
            )
            # logging.info(transitions)
            geominX, geomaxX, geominY, geomaxY = Utilities.geo_limits(geo_xml)

            print(
                "GeometrySize: X: ({:.2f},{:.2f}), Y: ({:.2f},{:.2f})".format(
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
                Utilities.plot_trajectories(
                    data, geometry_wall, transitions, geominX, geomaxX, geominY, geomaxY
                )
            else:
                Utilities.plot_trajectories(
                    data, geometry_wall, {}, geominX, geomaxX, geominY, geomaxY
                )

        if choose_dprofile:
            if data.shape[1] < 10 and how_speed == "from simulation":
                st.warning(
                    f"""trajectory file does not have enough columns ({data.shape[1]} < 10).
                \n Use <optional_output   speed=\"TRUE\">
                \n For more information refer to these links:
                """
                )
                st.code("https://www.jupedsim.org/jpscore_inifile.html#header")
                st.code(
                    "https://www.jupedsim.org/jpscore_trajectory.html#addtional-outputhttps://www.jupedsim.org/jpscore_inifile.html#header"
                )
                
            if how_speed == "from simulation":
                logging.info("speed by simulation")
                speed = data[:, 9]
            else:
                logging.info("speed by trajectory")
                speed = Utilities.compute_speed(data, fps, df)
 
            density = Utilities.weidmann(speed)
            pl.info(f"""
            Density in range [{np.min(density):.2f} : {np.max(density):.2f}] [1/m/m]\n
            Speed in range [{np.min(speed):.2} : {np.max(speed):.2}] [m/s]""")
            logging.info("plotting density profile")
            xbins = np.arange(geominX, geomaxX + dx, dx)
            ybins = np.arange(geominY, geomaxY + dx, dx)
            ret2 = stats.binned_statistic_2d(
                data[:, 2],
                data[:, 3],
                density,
                "mean",
                bins=[xbins, ybins],
            )
            prof2 = np.nan_to_num(ret2.statistic.T)
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(
                prof2,
                cmap=cm.jet,
                interpolation=interpolation,
                origin="lower",
                vmin=0,
                vmax=6, # np.max(density),
                extent=[geominX, geomaxX, geominY, geomaxY],
            )
            Utilities.plot_geometry(ax, geometry_wall)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3.5%", pad=0.3)
            cb = plt.colorbar(im, cax=cax)
            cb.set_label(r"$\rho\; / 1/m^2$", rotation=90, labelpad=15, fontsize=15)
            st.pyplot(fig)

        if choose_vprofile:
            if data.shape[1] < 10 and how_speed == "from simulation":
                st.warning(
                    f"""trajectory file does not have enough columns ({data.shape[1]} < 10).
                \n Use <optional_output   speed=\"TRUE\">
                \n For more information refer to these links:
                """
                )
                st.code("https://www.jupedsim.org/jpscore_inifile.html#header")
                st.code(
                    "https://www.jupedsim.org/jpscore_trajectory.html#addtional-outputhttps://www.jupedsim.org/jpscore_inifile.html#header"
                )
                st.stop()

            logging.info("plotting velocity profile")
            if how_speed == "from simulation":
                logging.info("speed by simulation")
                speed = data[:, 9]
            else:
                logging.info("speed by trajectory")
                speed = Utilities.compute_speed(data, fps, df)

            density = Utilities.weidmann(speed)
            pl.info(f"""
            Density in range [{np.min(density):.2f} : {np.max(density):.2f}] [1/m^2]\n
            Speed in range [{np.min(speed):.2} : {np.max(speed):.2}] [m/s]""")
            
            xbins = np.arange(geominX, geomaxX + dx, dx)
            ybins = np.arange(geominY, geomaxY + dx, dx)            
            ret = stats.binned_statistic_2d(
                data[:, 2], data[:, 3], speed, "mean", bins=[xbins, ybins]
            )
            prof = np.nan_to_num(ret.statistic.T)
            fig, ax = plt.subplots(1, 1)
            im = ax.imshow(
                prof,
                cmap=cm.jet.reversed(),
                interpolation=interpolation,
                origin="lower",
                vmin=0,
                vmax=1.3, #np.max(speed),
                extent=[geominX, geomaxX, geominY, geomaxY],
            )

            Utilities.plot_geometry(ax, geometry_wall)
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

            Utilities.plot_NT(stats, cum_num, fps)
            # --
            T = dt.datetime.now()
            n = trajectory_file.name.split(".txt")[0]
            file_download = f"{n}_{T.year}-{T.month:02}-{T.day:02}_{T.hour:02}-{T.minute:02}-{T.second:02}.txt"
            once = 1
            for i in selected_transitions:
                if once:
                    a = np.vstack((stats[i], cum_num[i]))
                    once = 0
                else:
                    a = np.vstack((a, stats[i], cum_num[i]))

            fmt = len(selected_transitions)*["%d", "%d"]
            a = a.T
            np.savetxt(
                file_download,
                a,
                fmt=fmt,
                header=np.array2string(np.array(selected_transitions, dtype=int),
                                       precision=2,
                                       separator='\t',
                                       suppress_small=True),
                comments="#",
                delimiter="\t",
            )
            with open(file_download, encoding='utf-8') as f:
                download = st.sidebar.download_button('Download statistics',
                                                      f,
                                                      file_name=file_download)
