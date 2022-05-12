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
        Trajectory column names: {self.header_traj}\n
        Unit: {self.unit}\n
        Frames per second: {self.fps}\n
        Frames: {len(frames)} | First: {frames[0]:.0f} | Last: {frames[-1]:.0f}\n
        Agents: {nagents}\n
        Evac-time: {np.max(frames)/self.fps} [s]
        """
        st.info(msg)
        st.markdown("### :chart_with_upwards_trend: Trajectories")
        with Utilities.profile("show_table"):
            logging.info(f"show table with {self.data.shape}")
            fig = plots.show_trajectories_table(self.data[:10, 0:5])
            st.plotly_chart(fig, use_container_width=True)
