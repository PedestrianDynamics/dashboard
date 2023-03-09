import sys

sys.path.append('../')
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp

import doc
import plots
#add an import to Hydralit
import Utilities


class ProfileClass(HydraHeadApp):
    def __init__(self, data, how_speed, geometry_wall, geominX, geomaxX, geominY, geomaxY, fps):
    
        self.how_speed = how_speed
        self.fps = fps
        self.data = data
        self.geominX = geominX
        self.geomaxX = geomaxX
        self.geominY = geominY
        self.geomaxY = geomaxY
        self.geometry_wall = geometry_wall
    


    def init_sidebar(self):
        #st.sidebar.header("🔴 Heatmaps")
        #prfx = st.sidebar.expander("Options")
        c1, c2 = st.sidebar.columns((1, 1))
        # choose_dprofile = c1.checkbox(
        #     "▶️ Show", help="Plot density and speed profiles", key="dProfile"
        # )
        # choose_vprofile = c2.checkbox("Speed", help="Plot speed profile", key="vProfile")
        choose_d_method = st.sidebar.radio(
            "Density method",
            ["Classical", "Gaussian", "Weidmann"],
            help="""
            How to calculate average of density over time and space""",
        )
        st.sidebar.write(
             "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
             unsafe_allow_html=True,
        )
        if choose_d_method == "Gaussian":
            width = st.sidebar.slider(
                "Width", 0.05, 1.0, 0.6, help="Width of Gaussian function"
            )
        else:
            width = 0.6

        dx = st.sidebar.slider("Grid size", 0.1, 4.0, 1.0, step=0.2, help="Space discretization")
        # methods = ["nearest", "gaussian", "sinc", "bicubic", "mitchell", "bilinear"]
        methods = ["off", "on"]
        interpolation = st.sidebar.radio(
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
        
        return choose_d_method, dx, width, interpolation



    
    def run(self):
        choose_d_method, dx, width, interpolation = ProfileClass.init_sidebar(self)
        frames = np.unique(self.data[:, 1])        
        info_profile = st.expander(
            "Documentation: Density/Speed maps (click to expand)"
        )
        with info_profile:
            doc.doc_profile()

        c1, c2 = st.columns((1, 1))
        Utilities.check_shape_and_stop(self.data.shape[1], self.how_speed)
        msg = ""
        if True:
            xbins = np.arange(self.geominX, self.geomaxX + dx, dx)
            ybins = np.arange(self.geominY, self.geomaxY + dx, dx)
            if choose_d_method == "Weidmann":
                density_ret = Utilities.calculate_density_average_weidmann(
                    self.geominX,
                    self.geomaxX,
                    self.geominY,
                    self.geomaxY,
                    dx,
                    dx,
                    len(frames),
                    self.data[:, 2],
                    self.data[:, 3],
                    self.data[:, st.session_state.speed_index],
                )
            elif choose_d_method == "Gaussian":
                with Utilities.profile("density profile gauss"):
                    density_ret = Utilities.calculate_density_average_gauss(
                        self.geominX,
                        self.geomaxX,
                        self.geominY,
                        self.geomaxY,
                        dx,
                        dx,
                        len(frames),
                        width,
                        self.data[:, 2],
                        self.data[:, 3],
                    )
            elif choose_d_method == "Classical":
                density_ret = np.zeros((len(ybins) - 1, len(xbins) - 1))
                density_ret = Utilities.calculate_density_average_classic(
                    self.geominX,
                    self.geomaxX,
                    self.geominY,
                    self.geomaxY,
                    dx,
                    dx,
                    len(frames),
                    self.data[:, 2],
                    self.data[:, 3],
                )
            st.session_state.density = density_ret
            msg += f"Density profile in range [{np.min(density_ret):.2f} : {np.max(density_ret):.2f}] [1/m^2]. \n"
            fig = plots.plot_profile_and_geometry2(
                xbins,
                ybins,
                self.geometry_wall,
                None,#st.session_state.xpos,
                None,#st.session_state.ypos,
                None,#st.session_state.lm,
                density_ret,
                interpolation,
                label=r"1/m/m",
                title="Density",
                vmin=None,
                vmax=None,
            )
            c1.plotly_chart(fig, use_container_width=True)
            if choose_d_method == "Gaussian":
                speed_ret = Utilities.weidmann(st.session_state.density)                    
            else:
                speed_ret = Utilities.calculate_speed_average(
                    self.geominX,
                    self.geomaxX,
                    self.geominY,
                    self.geomaxY,
                    dx,
                    dx,
                    len(frames),
                    self.data[:, 2],
                    self.data[:, 3],
                    self.data[:, st.session_state.speed_index],
                )

            fig = plots.plot_profile_and_geometry2(
                xbins,
                ybins,
                self.geometry_wall,
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
            c2.plotly_chart(fig, use_container_width=True)

            speed = self.data[:, st.session_state.speed_index]
            msg += f"Speed profile in range [{np.min(speed_ret):.2f} : {np.max(speed_ret):.2f}] [m/s]. "
            msg += f"Speed trajectory in range [{np.min(speed):.2f} : {np.max(speed):.2f}] [m/s]. "

            st.info(msg)

