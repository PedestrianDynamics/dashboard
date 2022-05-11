import sys

sys.path.append("../")
from shapely.geometry import LineString
import datetime as dt
import Utilities
import plots
import doc
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp
import logging


class JamClass(HydraHeadApp):
    def __init__(self, data, fps):
        self.data = data
        self.frames = np.unique(self.data[:, 1])
        self.peds = np.unique(data[:, 0]).astype(int)
        self.fps = fps

    def init_sidebar(self):
        st.sidebar.header("🐌 Jam")
        # choose_jam_duration = st.sidebar.checkbox(
        # "▶️ Show",
        # value=False,
        # help="Plot change of the number of pedestrian in jam versus time",
        # key="jam_duration",
        # )
        jam_speed = st.sidebar.slider(
            "Min jam speed / m/s",
            0.1,
            1.0,
            0.5,
            help="An agent slower that this speed is in jam",
            key="jVmin",
        )
        min_jam_time = st.sidebar.slider(
            "Min jam duration / s",
            1,
            180,
            1,
            help="A jam lasts at least that long",
            key="jTmin",
        )
        min_jam_agents = st.sidebar.slider(
            "Min agents in jam",
            2,
            200,
            20,
            help="A jam has at least so many agents",
            key="jNmin",
        )

        return jam_speed, min_jam_time, min_jam_agents

    def run(self):
        info = st.expander("Documentation: Jam definitions (click to expand)")
        with info:
            doc.doc_jam()

        jam_speed, min_jam_time, min_jam_agents = JamClass.init_sidebar(self)
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
            int(10 * self.fps),
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
        jam_frames = Utilities.jam_frames(self.data, jam_speed)
        with Utilities.profile("jam_lifetime"):
            lifetime, chuncks, max_lifetime, from_to = Utilities.jam_lifetime(
                self.data, jam_frames[10:], min_jam_agents, self.fps, precision
            )  # remove the first frames, cause in simulation people stand

        ## duration
        logging.info(f"waiting time with {min_jam_time}")
        with Utilities.profile("jam_waiting_time"):
            waiting_time = Utilities.jam_waiting_time(
                self.data, jam_speed, min_jam_time, self.fps, precision
            )

        if not waiting_time.size:
            wtimes = np.array([])
        else:
            wtimes = waiting_time[:, 1]

        with Utilities.profile("Rendering Jam figures"):
            ## plots
            fig1 = plots.plot_jam_lifetime(
                self.frames,
                lifetime,
                self.fps,
                max_lifetime,
                from_to,
                min_jam_agents,
            )
            hist = plots.plot_jam_lifetime_hist(chuncks, self.fps, nbins)
            pl2.plotly_chart(fig1, use_container_width=True)
            pl.plotly_chart(hist, use_container_width=True)
            # --
            hist = plots.plot_jam_waiting_hist(wtimes, self.fps, nbins2)
            pl3.plotly_chart(hist, use_container_width=True)
