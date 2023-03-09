import sys

sys.path.append("../")
import logging

import numpy as np
import plots
import streamlit as st
import Utilities
from hydralit import HydraHeadApp


class StatClass(HydraHeadApp):
    def __init__(self, data, unit, fps, header_traj):
        self.unit = unit
        self.fps = fps
        self.data = data
        self.header_traj = header_traj

    def run(self):
        st.markdown("### :round_pushpin: Summary of the trajectory data")
        frames = np.unique(self.data[:, 1])
        peds = np.unique(self.data[:, 0]).astype(int)
        nagents = len(peds)
        msg = f"""
        Header: {self.header_traj}\n
        Unit: {self.unit}\n
        Frames per second: {self.fps}\n
        """
        c1, c2, c3 = st.columns((1, 1, 1))
        c1.metric(label="Agents: ", value=nagents)
        c2.metric("Time: ", f"{np.max(frames) / self.fps:.2f} [s]")
        c3.metric(label="Frames: ", value=len(frames), delta=int(frames[-1]))
        st.info(msg)

        st.markdown("### :chart_with_upwards_trend: Trajectories")
        with Utilities.profile("show_table"):
            logging.info(f"show table with {self.data.shape}")
            fig = plots.show_trajectories_table(self.data[:10, 0:5])
            st.plotly_chart(fig, use_container_width=True)
