import sys

sys.path.append('../')
#add an import to Hydralit
import Utilities
import plots
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp
import logging

class StatClass(HydraHeadApp):
    def __init__(self, data, unit, fps, header_traj):
        self.unit = unit
        self.fps = fps
        self.data = data
        self.header_traj = header_traj


    def run(self):
        st.markdown("### :bar_chart: Summary of the data")
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
        with Utilities.profile("show_table"):
            logging.info(f"show table with {self.data.shape}")
            fig = plots.show_trajectories_table(self.data[:10, 0:5])
            st.plotly_chart(fig, use_container_width=True)
            
        
