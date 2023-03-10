import sys

sys.path.append("../")
import collections
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from hydralit import HydraHeadApp
from streamlit_drawable_canvas import st_canvas

import doc
import draw_geometry as dg
import plots

# add an import to Hydralit
import Utilities


class dvTimeSeriesClass(HydraHeadApp):
    def __init__(
        self,
        title,
        data,
        how_speed,
        geometry_wall,
        geominX,
        geomaxX,
        geominY,
        geomaxY,
        fps,
        newdata,
    ):
        self.title = title
        self.how_speed = how_speed
        self.fps = fps
        self.data = data
        self.geominX = geominX
        self.geomaxX = geomaxX
        self.geominY = geominY
        self.geomaxY = geomaxY
        self.geometry_wall = geometry_wall
        self.newdata = newdata

    def init_sidebar(self):
        logging.info(f"newdata {self.newdata}")
        frames = np.unique(self.data[:, 1])
        choose_d_method = st.sidebar.radio(
            "Density method",
            ["Classic", "Gaussian"],
            help="""
            How to calculate average of density over time and space""",
        )
        st.sidebar.write(
            "<style>div.row-widget.stRadio > div{flex-direction:row;}</style>",
            unsafe_allow_html=True,
        )
        sample = st.sidebar.slider(
            "Sample rate",
            min_value=2,
            max_value=int(np.max(frames * 0.2)),
            value=10,
            step=5,
            help="Sample rate of ploting trajectories and time series \
                (the lower the slower)",
            key="sample_traj",
        )

        if choose_d_method == "Gaussian":
            gauss_width = st.sidebar.slider(
                "Gauss width", 0.05, 1.0, 0.6, help="Width of Gaussian function"
            )
        else:
            gauss_width = 0.6

        drawing_mode = st.sidebar.radio(
            #    "Drawing tool:", ("Area", "Line", "polygon", "Transform"))
            "Measurement:",
            ("Area", "Transform"),
        )

        if drawing_mode in ["wall", "Line"]:
            drawing_mode = "line"

        if drawing_mode == "Area":
            drawing_mode = "rect"

        if drawing_mode == "Transform":
            drawing_mode = "transform"

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

        # -------
        if st.session_state.bg_img is None:
            logging.info("START new canvas")
            bg_img, img_width, img_height, dpi, scale = dvTimeSeriesClass.bg_img(self)
            st.session_state.scale = scale
            st.session_state.dpi = dpi
            st.session_state.img_width = img_width
            st.session_state.img_height = img_height
            st.session_state.bg_img = bg_img
        else:
            bg_img = st.session_state.bg_img
            scale = st.session_state.scale
            dpi = st.session_state.dpi
            img_height = st.session_state.img_height
            img_width = st.session_state.img_width

        if "canvas" in globals():
            del canvas

        canvas = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color="#060EE8",
            background_color="#eee",
            background_image=bg_img,
            update_streamlit=True,
            width=img_width,
            height=img_height,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        return sample, choose_d_method, gauss_width, canvas, dpi, scale, img_height

    def draw_rects(self, canvas, img_height, dpi, scale):
        rect_points_xml = collections.defaultdict(dict)
        if canvas.json_data is not None:
            objects = pd.json_normalize(canvas.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")

            if not objects.empty:
                rects = objects[objects["type"].values == "rect"]
                if not rects.empty:
                    (
                        rfirst_x,
                        rfirst_y,
                        rsecond_x,
                        rsecond_y,
                        rthird_x,
                        rthird_y,
                        rfirth_x,
                        rfirth_y,
                    ) = dg.process_rects(rects, img_height)
                    i = 0
                    for x1, x2, x3, x4, y1, y2, y3, y4 in zip(
                        rfirst_x,
                        rsecond_x,
                        rthird_x,
                        rfirth_x,
                        rfirst_y,
                        rsecond_y,
                        rthird_y,
                        rfirth_y,
                    ):
                        rect_points_xml[i]["x"] = [
                            x1 / scale / dpi + self.geominX,
                            x2 / scale / dpi + self.geominX,
                            x3 / scale / dpi + self.geominX,
                            x4 / scale / dpi + self.geominX,
                        ]
                        rect_points_xml[i]["y"] = [
                            y1 / scale / dpi + self.geominY,
                            y2 / scale / dpi + self.geominY,
                            y3 / scale / dpi + self.geominY,
                            y4 / scale / dpi + self.geominY,
                        ]
                        i += 1

        return rect_points_xml

    def bg_img(self):
        logging.info("enter bg_img")
        width, height, scale = dg.get_scaled_dimensions(
            self.geominX, self.geomaxX, self.geominY, self.geomaxY
        )
        fig, ax = plt.subplots(figsize=(width, height))
        fig.set_dpi(100)
        ax.set_xlim((0, width))
        ax.set_ylim((0, height))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        inv = ax.transData.inverted()
        dg.plot_traj(ax, self.data, scale, self.geominX, self.geominY)
        major_ticks_top_x = np.linspace(0, width, 5)
        major_ticks_top_y = np.linspace(0, height, 5)
        minor_ticks_top_x = np.linspace(0, width, 40)
        minor_ticks_top_y = np.linspace(0, height, 40)
        major_ticks_bottom_x = np.linspace(0, width, 20)
        major_ticks_bottom_y = np.linspace(0, height, 20)
        ax.set_xticks(major_ticks_top_x)
        ax.set_yticks(major_ticks_top_y)
        ax.set_xticks(minor_ticks_top_x, minor=True)
        ax.set_yticks(minor_ticks_top_y, minor=True)
        ax.grid(which="major", alpha=0.6)
        ax.grid(which="minor", alpha=0.3)
        ax.set_xticks(major_ticks_bottom_x)
        ax.set_yticks(major_ticks_bottom_y)
        ax.grid()
        bg_img = dg.fig2img(fig)
        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        img_width, img_height = bbox.width * fig.dpi, bbox.height * fig.dpi
        # inv = ax.transData.inverted()
        return bg_img, img_width, img_height, fig.dpi, scale

    def run(self):
        info_timeseries = st.expander(
            "Documentation: Density/Speed Time Series (click to expand)"
        )
        st.info(
            "Draw a measurement area with **Area** and move it if necessary with **Transform**"
        )
        (
            sample,
            choose_d_method,
            gauss_width,
            canvas,
            dpi,
            scale,
            img_height,
        ) = dvTimeSeriesClass.init_sidebar(self)
        # canvas

        with info_timeseries:
            doc.doc_timeseries()

        frames = np.unique(self.data[:, 1])
        rects = dvTimeSeriesClass.draw_rects(self, canvas, img_height, dpi, scale)
        for ir in range(len(rects)):
            pl = st.empty()
            c1, c2, c3 = st.columns((1, 1, 1))
            _, _, c31 = st.columns((1, 1, 1))
            pl_l = c31.empty()
            # st.write((rects[ir]['x'][0], rects[ir]['x'][1], rects[ir]['x'][2], rects[ir]['x'][3]))
            # st.write((rects[ir]['y'][0], rects[ir]['y'][1], rects[ir]['y'][2], rects[ir]['y'][3]))
            from_x = rects[ir]["x"][0]
            to_x = rects[ir]["x"][1]
            from_y = rects[ir]["y"][3]
            to_y = rects[ir]["y"][0]
            dx = to_x - from_x
            dy = to_y - from_y
            pl.info(f"Measurement area {ir+1}, dx = {dx:.2f} / m, dy = {dy:.2f} / m")
            if choose_d_method == "Gaussian":
                with Utilities.profile("time series gauss"):
                    density_time = []
                    for frame in frames[::sample]:
                        dframe = self.data[:, 1] == frame
                        x = self.data[dframe][:, 2]
                        y = self.data[dframe][:, 3]
                        dtime = Utilities.calculate_density_average_gauss(
                            from_x,
                            to_x,
                            from_y,
                            to_y,
                            dx,
                            dy,
                            1,
                            gauss_width,
                            x,
                            y,
                        )
                        density_time.append(dtime[0, 0])

                speed_time = Utilities.weidmann(np.array(density_time))

            if choose_d_method == "Classic":
                with Utilities.profile("time series classic"):
                    density_time = []
                    for frame in frames[::sample]:
                        dframe = self.data[:, 1] == frame
                        x = self.data[dframe][:, 2]
                        y = self.data[dframe][:, 3]
                        dtime = Utilities.calculate_density_frame_classic(
                            from_x,
                            to_x,
                            from_y,
                            to_y,
                            dx,
                            dy,
                            x,
                            y,
                        )
                        # st.write(dtime)
                        density_time.append(dtime[0, 0])

                    speed_time = []
                    for frame in frames[::sample]:
                        dframe = self.data[:, 1] == frame
                        x = self.data[dframe][:, 2]
                        y = self.data[dframe][:, 3]
                        speed_agent = self.data[dframe][:, st.session_state.speed_index]
                        stime = Utilities.calculate_speed_average(
                            from_x,
                            to_x,
                            from_y,
                            to_y,
                            dx,
                            dy,
                            x,
                            y,
                            speed_agent,
                        )
                        speed_time.append(stime[0, 0])

            # ---- plots
            # rho
            fig = plots.plot_timeserie(
                frames,
                density_time,
                self.fps,
                "Density / m / m",
                np.min(density_time),
                np.max(density_time) + 1,
                np.max(density_time),
            )
            c2.plotly_chart(fig, use_container_width=True)
            # v
            fig = plots.plot_timeserie(
                frames,
                speed_time,
                self.fps,
                "Speed / m/s",
                np.min(speed_time),
                np.max(speed_time) + 1,
                np.max(speed_time),
            )
            c1.plotly_chart(fig, use_container_width=True)
            # Js
            l = pl_l.slider(
                "length",
                float(np.min((dx, dy))),
                float(np.max((dx, dy))),
                0.5,
                dx / 10,
                help="flow = rho * v / length",
            )
            flow = np.array(density_time) * np.array(speed_time) / l
            fig = plots.plot_timeserie(
                frames,
                flow,
                self.fps,
                "Js / 1/s",
                np.min(flow),
                np.max(flow) + 1,
                np.max(flow),
            )

            c3.plotly_chart(fig, use_container_width=True)
