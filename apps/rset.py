import sys

sys.path.append('../')
#add an import to Hydralit
import Utilities
import plots
import doc
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp


class RsetClass(HydraHeadApp):
    def __init__(self, data, geominX, geomaxX, geominY, geomaxY, geometry_wall, fps):
        self.fps = fps
        self.data = data
        self.geominX = geominX
        self.geomaxX = geomaxX
        self.geominY = geominY
        self.geomaxY = geomaxY
        self.geometry_wall = geometry_wall


    def init_sidebar(self):
        #st.sidebar.header("🔴 RSET")
        dx = st.sidebar.slider("Grid size", 0.1, 4.0, 1.0, step=0.2, help="Space discretization")
        methods = ["off", "on"]
        
        interpolation = st.sidebar.radio(
            "Smooth", methods, help="Smoothen the heatmaps"
        )
        if interpolation == "off":
            interpolation = "false"
        else:
            interpolation = "best"

        return dx, interpolation


    def run(self):
        info_rset = st.expander(
            "Documentation: RSET maps (click to expand)"
        )
        with info_rset:
            doc.doc_RSET()

        c1, c2 = st.columns((1, 1))
        pr1 = c2.empty()
        pr2 = c1.empty()
        
        dx, interpolation = RsetClass.init_sidebar(self)
        
        
        
        
        xbins = np.arange(self.geominX, self.geomaxX + dx, dx)
        ybins = np.arange(self.geominY, self.geomaxY + dx, dx)
        # RSET
        rset_max = Utilities.calculate_RSET(self.geominX,
                                            self.geomaxX,
                                            self.geominY,
                                            self.geomaxY,
                                            dx,
                                            dx,
                                            self.data[:, 2],
                                            self.data[:, 3],
                                            self.data[:, 1]/self.fps,
                                            "max")

        
        nbins = c2.slider(
            "Number of bins", 5, 40, value=10, help="Number of bins", key="rset_bins"
        )
        
        fig = plots.plot_RSET_hist(rset_max, nbins)
        
        pr1.plotly_chart(fig, use_container_width=True)

        fig = plots.plot_profile_and_geometry2(
            xbins,
            ybins,
            self.geometry_wall,
            None, None, None,
            rset_max,
            interpolation,
            label=r"time / s",
            title=f"RSET = {np.max(rset_max):.1f} / s",
            vmin=None,
            vmax=None,
        )
        pr2.plotly_chart(fig, use_container_width=True)
