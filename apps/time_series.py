import sys
from shapely.geometry import LineString
import datetime as dt

sys.path.append("../")
# add an import to Hydralit
import Utilities
import plots
import doc
import numpy as np
import streamlit as st
from hydralit import HydraHeadApp
import logging


class TimeSeriesClass(HydraHeadApp):
    def __init__(self, data, disable_NT_flow, transitions, default, fps, name, group_index):
        self.data = data
        self.disable_NT_flow = disable_NT_flow
        self.frames = np.unique(self.data[:, 1])
        self.peds = np.unique(data[:, 0]).astype(int)
        self.transitions = transitions
        self.default = default
        self.fps = fps
        self.name = name
        self.group_index = group_index

    def init_sidebar(self):
        st.sidebar.header("📊 Summary curves")
        c1, c2 = st.sidebar.columns((1, 1))
        choose_NT = c1.checkbox(
            "N-T",
            value=True,
            help="Plot N-t curve",
            key="NT",
            disabled=self.disable_NT_flow,
        )
        choose_flow = c2.checkbox(
            "Flow",
            value=True,
            help="Plot flow curve",
            key="Flow",
            disabled=self.disable_NT_flow,
        )
        choose_time_distance = c1.checkbox(
            "T-D",
            value=True,
            help="Plot Time-Distance to the first selected entrance",
            key="EvacT",
            disabled=self.disable_NT_flow,
        )
        choose_survival = c2.checkbox(
            "Survival",
            value=True,
            help="Plot survival function (clogging)",
            disabled=self.disable_NT_flow,
            key="Survival",
        )
        num_peds_TD = st.sidebar.number_input(
            "number pedestrians",
            min_value=1,
            max_value=len(self.peds),
            value=int(0.3 * len(self.peds)),
            step=1,
            help="number of pedestrians to show in T-D",
        )
        sample_TD = st.sidebar.number_input(
            "sample",
            min_value=1,
            max_value=int(0.1 * len(self.frames)),
            value=int(0.01 * len(self.frames)),
            step=1,
            help="sample rate in T-D",
        )

        selected_transitions = st.sidebar.multiselect(
            "Select transition",
            self.transitions.keys(),
            self.default,
            help="Transition to calculate N-T. Can select multiple transitions",
        )

        return (
            selected_transitions,
            choose_NT,
            choose_flow,
            choose_time_distance,
            choose_survival,
            num_peds_TD,
            sample_TD,
        )

    def run(self):
        (
            selected_transitions,
            choose_NT,
            choose_flow,
            choose_time_distance,
            choose_survival,
            num_peds_TD,
            sample_TD,
        ) = TimeSeriesClass.init_sidebar(self)
        plot_options = (
            choose_NT or choose_flow or choose_time_distance or choose_survival
        )
        info = st.expander("Documentation: Plot curves (click to expand)")
        with info:
            doc.doc_plots()

        # all these options need to calculate N-T-Data
        if plot_options:
            # todo: cache calculation in st.session_state.tstats
            with Utilities.profile("calculate_NT_data"):
                (
                    tstats,
                    cum_num,
                    cum_num_positiv,
                    cum_num_negativ,
                    trans_used,
                    max_len,
                    msg,
                ) = Utilities.calculate_NT_data(
                    self.transitions,
                    selected_transitions,
                    self.data,
                    self.fps,
                )

        c1, c2, c3 = st.columns((1, 1, 1))
        if choose_NT:
            peds_inside = Utilities.peds_inside(self.data)
            fig1 = plots.plot_peds_inside(self.frames, peds_inside, self.fps)
            c2.plotly_chart(fig1, use_container_width=True)
            if tstats:
                fig2 = plots.plot_NT(
                    tstats, cum_num, cum_num_positiv, cum_num_negativ, self.fps
                )
                c1.plotly_chart(fig2, use_container_width=True)

            if choose_flow and tstats:
                fig = plots.plot_flow(
                    tstats, cum_num, cum_num_positiv, cum_num_negativ, self.fps
                )
                c3.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns((1, 1))
        if choose_time_distance:
            with c1:
                with Utilities.profile("plot distance-time curve"):
                    selected_and_used_transitions = [
                        i for i in selected_transitions if trans_used[i]
                    ]
                    if selected_and_used_transitions:
                        i = selected_and_used_transitions[0]
                        Frames = tstats[i]
                        fig = plots.plot_time_distance(
                            Frames,
                            self.data,
                            LineString(self.transitions[i]),
                            i,
                            self.fps,
                            num_peds_TD,
                            sample_TD,
                            self.group_index
                        )
                        st.plotly_chart(fig, use_container_width=True)

        if choose_survival:
            with c2:
                if selected_transitions:
                    fig = plots.plot_survival(tstats, self.fps)
                    st.plotly_chart(fig, use_container_width=True)

        if selected_transitions:
            st.info(msg)

        # -- download stats
        if choose_NT:
            T = dt.datetime.now()

            file_download = f"stats_{self.name}_{T.year}-{T.month:02}-{T.day:02}_{T.hour:02}-{T.minute:02}-{T.second:02}.txt"
            once = 0  # don't download if file is empty
            for i in selected_transitions:
                if not trans_used[i]:
                    continue

                nrows = tstats[i].shape[0]
                if nrows < max_len:
                    tmp_stats = np.full((max_len, 3), -1)
                    tmp_stats[:nrows, :] = tstats[i]
                    tmp_cum_num = np.full(max_len, -1)
                    tmp_cum_num[: len(cum_num[i])] = cum_num[i]
                else:
                    tmp_stats = tstats[i]
                    tmp_cum_num = cum_num[i]

                tmp_cum_num_p = np.full(len(tmp_cum_num), -1)
                tmp_cum_num_p[: len(cum_num_positiv[i])] = cum_num_positiv[i]
                tmp_cum_num_n = np.full(len(tmp_cum_num), -1)
                tmp_cum_num_n[: len(cum_num_negativ[i])] = cum_num_negativ[i]

                tmp_cum_num = tmp_cum_num.reshape(len(tmp_cum_num), 1)
                tmp_cum_num_p = tmp_cum_num_p.reshape(len(tmp_cum_num_p), 1)
                tmp_cum_num_n = tmp_cum_num_n.reshape(len(tmp_cum_num_n), 1)
                if not once:
                    all_stats = np.hstack(
                        (tmp_stats, tmp_cum_num, tmp_cum_num_p, tmp_cum_num_n)
                    )
                    once = 1
                else:
                    all_stats = np.hstack(
                        (
                            all_stats,
                            tmp_stats,
                            tmp_cum_num,
                            tmp_cum_num_p,
                            tmp_cum_num_n,
                        )
                    )

            if selected_transitions and once:
                passed_lines = [i for i in selected_transitions if trans_used[i]]
                fmt = len(passed_lines) * ["%d", "%d", "%d", "%d", "%d", "%d"]
                # all_stats = all_stats.T
                np.savetxt(
                    file_download,
                    all_stats,
                    fmt=fmt,
                    header="line ids: \n"
                    + np.array2string(
                        np.array(passed_lines, dtype=int),
                        precision=2,
                        separator="\t",
                        suppress_small=True,
                    )
                    + f"\nframerate: {self.fps:.0f}"
                    + "\npid\tframe\tdirection\tcount_tot\tcount+\tcount-",
                    comments="#",
                    delimiter="\t",
                )
                with open(file_download, encoding="utf-8") as f:
                    st.sidebar.download_button(
                        "Download statistics", f, file_name=file_download
                    )
