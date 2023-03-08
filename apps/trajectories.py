import sys

sys.path.append("../")
import Utilities
import plots
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp


class TrajClass(HydraHeadApp):
    def __init__(
        self,
        data,
        data_df,
        how_speed,
        geometry_wall,
        transitions,
        geominX,
        geomaxX,
        geominY,
        geomaxY,
        fps,
    ):
        self.show_special_agent_stats = False
        self.plot_ped = -1
        self.how_speed = how_speed
        self.fps = fps
        self.data = data
        self.data_df = data_df
        self.geominX = geominX
        self.geomaxX = geomaxX
        self.geominY = geominY
        self.geomaxY = geomaxY
        self.geometry_wall = geometry_wall
        self.choose_transitions = True
        self.transitions = transitions
        self.sample_trajectories = 1
        self.frames = np.unique(self.data[:, 1])
        self.peds = np.unique(data[:, 0]).astype(int)
        nagents = len(self.peds)

        if nagents <= 10:
            self.special_agent = 5
        else:
            self.special_agent = self.peds[10]

    def init_sidebar(self):
        st.sidebar.header("📉 Plot trajectories")
        c1, c2 = st.sidebar.columns((1, 1))
        self.choose_transitions = c2.checkbox(
            "Transitions", help="Show transittions", value=True, key="Tran"
        )
        self.show_special_agent_stats = c1.checkbox(
            "Agent",
            help="Show speed/angle and trajectory of highlighted agent",
            value=False,
            key="SpecialAgent",
        )
        self.choose_visualisation = c1.checkbox(
            "Animation", help="Show visualisation", value=False, key="Vis"
        )

        sample_trajectories = st.sidebar.number_input(
            "Sample rate",
            min_value=1,
            max_value=int(np.max(self.frames * 0.2)),
            value=10,
            step=5,
            help="Sample rate of ploting trajectories and time series \
                (the lower the slower)",
            key="sample_traj",
        )
        # hack. I dont know why this is sometimes float
        self.sample_trajectories = int(sample_trajectories)
        plot_ped = st.sidebar.number_input(
            "Highlight pedestrian",
            min_value=np.min(self.peds),
            max_value=np.max(self.peds),
            value=self.special_agent,
            step=10,
            help="Choose a pedestrian by id",
        )
        self.plot_ped = plot_ped

    def run(self):
        TrajClass.init_sidebar(self)
        sample_trajectories = self.sample_trajectories
        if self.show_special_agent_stats:
            agent = self.data[self.data[:, 0] == self.plot_ped]
            speed_agent = agent[:, st.session_state.speed_index]
            if self.how_speed == "from simulation":
                angle_agent = agent[:, 7]
            else:
                angle_agent = agent[:, -2]
        else:
            agent = []
            speed_agent = []
            angle_agent = []
            self.plot_ped = -1
            sample_trajectories = 1

        c1, c2 = st.columns((1, 1))
        with Utilities.profile("plot_trajectories"):
            fig = plots.plot_trajectories(
                self.data,
                self.data_df,
                self.plot_ped,
                speed_agent,
                self.geometry_wall,
                self.transitions,
                self.geominX,
                self.geomaxX,
                self.geominY,
                self.geomaxY,
                self.choose_transitions,
                sample_trajectories,
            )
            st.plotly_chart(fig, use_container_width=True)

        if self.choose_visualisation:
            with Utilities.profile("vis_trajectories"):
                fig = plots.moving_trajectories(
                    self.data,
                    self.data_df,
                    self.plot_ped,
                    speed_agent,
                    self.geometry_wall,
                    self.transitions,
                    self.geominX,
                    self.geomaxX,
                    self.geominY,
                    self.geomaxY,
                    self.choose_transitions,
                    sample_trajectories,
                )
                st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns((1, 1, 1))
        if self.show_special_agent_stats:
            with Utilities.profile("plot_agent_xy"):
                fig = plots.plot_agent_xy(
                    agent[::sample_trajectories, 1],
                    agent[::sample_trajectories, 2],
                    agent[:, 3],
                    self.fps,
                )
                c1.plotly_chart(fig, use_container_width=True)

            with Utilities.profile("plot_agent_angle"):
                fig = plots.plot_agent_angle(
                    self.plot_ped,
                    agent[::sample_trajectories, 1],
                    angle_agent[::sample_trajectories],
                    self.fps,
                )
                c2.plotly_chart(fig, use_container_width=True)

            with Utilities.profile("plot_agent_speed"):

                fig = plots.plot_agent_speed(
                    self.plot_ped,
                    agent[::sample_trajectories, 1],
                    speed_agent[::sample_trajectories],
                    np.max(self.data[:, st.session_state.speed_index]),
                    self.fps,
                )
                c3.plotly_chart(fig, use_container_width=True)
