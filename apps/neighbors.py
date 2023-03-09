import sys

sys.path.append("../")
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp

import doc
import plots
# add an import to Hydralit
import Utilities


class NeighborsClass(HydraHeadApp):
    def __init__(self, data, geominX, geomaxX, geominY, geomaxY, geometry_wall):
        # self.fps = fps
        self.data = data
        self.frames = np.unique(self.data[:, 1])
        self.geominX = geominX
        self.geomaxX = geomaxX
        self.geominY = geominY
        self.geomaxY = geomaxY
        self.geo_walls = geometry_wall
        self.peds = np.unique(self.data[:, 0]).astype(int)
        self.single_file = False

    def init_sidebar(self):
        st.header(f"Settings")
        c00, c01, c02, c03 = st.columns((0.4, 1, 1, 1))
        self.single_file = c00.checkbox("Single file?", help="Choose if trajectory is single-file movement")
        agent = c01.number_input(
            "Agent",
            min_value=(np.min(self.peds)),
            max_value=int(np.max(self.peds)),
            value=int((np.min(self.peds) + np.max(self.peds)) / 2),
            step=1,
            help="See data of this Agent",
        )

        dd = self.data[self.data[:, 0] == agent][:, 1]        
        frame = c03.slider(
            "Frame",
            int(np.min(dd)),
            int(np.max(dd)),
            int((np.min(dd) + np.max(dd)) / 2),
            step=1,
            help="Frame at which to display a snapshot of the neighbors",
        )

        k = c02.number_input(
            "K-neighbors",
            min_value=1,
            max_value=10,
            value=6,
            step=1,
            help="K-neighbors",
        )

        return (
            int(frame),
            int(agent),
            int(k),
        )  # todo: k should be int, but somehow it is float! weird!

    def run(self):
        info_rset = st.expander(
            "Documentation: K-Nearest-Neighbors' Analysis  (click to expand)"
        )
        # todo
        with info_rset:
            doc.doc_neighbors()

        frame, agent, k = NeighborsClass.init_sidebar(self)
        # placement of plots
        st.header(f"Data for pedestrian {agent}")
        c1, c2, c3 = st.columns((1, 1, 1))
        
        pr1 = c1.empty()
        pr2 = c2.empty()
        pr3 = c3.empty()
        st.header(f"Data for all pedestrians")
        c3, c4, c5 = st.columns((1, 1, 1))
        pr4 = c3.empty()
        pr5 = c4.empty()
        pr6 = c5.empty()
        k += 1
        if self.single_file:
            k = 2

        nearest_dist, nearest_ind = Utilities.get_neighbors_at_frame(
            frame, self.data, k
        )

        pdf_At_frame = Utilities.get_neighbors_pdf(nearest_dist[:, 1:].flatten())
        neighbors, neighbors_ids, area, agent_distances, agent_speeds = Utilities.get_neighbors_special_agent_data(
            agent, frame, self.data, nearest_dist, nearest_ind
        )        
        areas = []
        Speeds = np.array([])
        frames_areas = []
        distances = np.array([])
        C = np.array([]) # Qu2020a Eq. (4)
        c0 = np.exp(-1.5)
        for fr in self.frames:
            nearest_dist0, nearest_ind0 = Utilities.get_neighbors_at_frame(
                fr, self.data, k
            )
            at_frame = self.data[self.data[:, 1] == frame]
            Ids = at_frame[:, 0]            
            if nearest_dist0.shape[0] > 2:
                _neighbors, _ids, area, agent_dists, agent_speeds = Utilities.get_neighbors_special_agent_data(
                    agent, fr, self.data, nearest_dist0, nearest_ind0
                )                
                C = np.hstack((C, np.sum(np.exp(-agent_dists))))
                areas.append(area)
                frames_areas.append(fr)
             
                distances = np.hstack(
                    (distances, np.mean(nearest_dist0[:, 1:].flatten()))                
                )  # skip first column which is zeros.
                Speeds = np.hstack(
                    (Speeds, agent_speeds.flatten())                
                )  # skip first column which is zeros.

                
        # with open('test.npy', 'wb') as f:
        #     np.save(f, C)
        
        # plots
        if not self.single_file:
            fig = plots.plot_areas(areas, frames_areas, agent)
        else:
            fig = plots.plot_x_y(
                Speeds,
                distances,
                title=f"Headway-Speed for agent {agent}",
                xlabel="Speed / m/s",
                ylabel="Dist / m"
            )
        pr2.plotly_chart(fig, use_container_width=True)
        
        # plots for special agent
        fig = plots.plot_agents(
            agent,
            frame,
            self.data,
            neighbors,
            self.geo_walls,
            self.geominX,
            self.geomaxX,
            self.geominY,
            self.geomaxY,
        )
        pr1.plotly_chart(fig, use_container_width=True)
        
        # pdf of distances
        fig = plots.plot_x_y(
            nearest_dist,
            pdf_At_frame,
            title=f"PDF of distances at frame {frame}. Mean = {np.mean(nearest_dist):.2f} m",
            xlabel="Distance / m",
            ylabel="PDF"
        )
        
        fig = plots.plot_x_y(frames_areas,
                             C,
                             title=f"Contact index. Mean = {np.mean(C):.2f} m. Min = {np.min(C):.2f} m, Max = {np.max(C) :.2f} m",
                             xlabel="Frame",
                             ylabel="C / m",
                             threshold=c0)
        pr3.plotly_chart(fig, use_container_width=True)

        # plots for all pedestrians
        fig = plots.plot_x_y(
            nearest_dist,
            pdf_At_frame,
            title=f"PDF of distances at frame {frame}. Mean = {np.mean(nearest_dist):.2f} m",
            xlabel="Distance / m",
            ylabel="PDF"
        )
        pr4.plotly_chart(fig, use_container_width=True)
        
        pdf = Utilities.get_neighbors_pdf(distances)
        fig = plots.plot_x_y(distances,
                             pdf,
                             title=f"PDF of all distances for all frames. Mean = {np.mean(distances):.2f} m",
                             xlabel="Dist / m",
                             ylabel="PDF")
        pr5.plotly_chart(fig, use_container_width=True)

        fig = plots.plot_x_y(frames_areas[::10],
                             distances[::10],
                             title=f"Distances for all frames. Mean = {np.mean(distances):.2f} m. Min = {np.min(distances):.2f} m, Max = {np.max(distances) :.2f} m",
                             xlabel="Frame",
                             ylabel="Distance / m")
        pr6.plotly_chart(fig, use_container_width=True)
 


        
