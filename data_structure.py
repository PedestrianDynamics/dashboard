import sys

sys.path.append(".")
import logging
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Any, List
from xml.dom.minidom import parse, parseString, Document
import pandas as pd
import numpy as np
import numpy.typing as npt
import streamlit as st
from pandas import read_csv

import Utilities


@dataclass
class data_files:
    """
    Class handling trajectory and geometry files
    """

    uploaded_traj_file: BytesIO
    uploaded_geo_file: BytesIO
    from_examples: str
    traj_name: str = field(init=False, default="")
    selected_traj_file: str = field(init=False, default="")
    selected_geo_file: str = field(init=False, default="")
    got_traj_data: Any = field(init=False, default=False)
    _data: npt.NDArray[np.float32] = field(init=False, default=np.array([]))
    _df: pd.DataFrame = field(init=False)
    default_geometry_file: str = (
        "geometry.xml"  # in case trajectories have no geometry files
    )

    def process_traj_file(self) -> str:
        """return StringIO data from trajectory file"""
        if self.uploaded_traj_file:
            stringio = StringIO(self.uploaded_traj_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
        else:
            with open(self.selected_traj_file, encoding="utf-8") as f:
                string_data = f.read()

        logging.info("got some data")
        return string_data

    def process_geo_file(self) -> str:
        """return data from geometry file"""
        if self.uploaded_geo_file is None:
            with open(
                self.default_geometry_file, encoding="utf-8"
            ) as geometry_file_obj:
                geo_string_data = geometry_file_obj.read()

        else:
            geo_stringio = StringIO(self.uploaded_geo_file.getvalue().decode("utf-8"))
            geo_string_data = geo_stringio.read()

        if self.selected_geo_file:
            with open(self.selected_geo_file, encoding="utf-8") as geometry_file_obj:
                geo_string_data = geometry_file_obj.read()

        return geo_string_data

    def read_traj_data(self):
        """Set _data with trajectories if traj file uploaded or selected"""
        logging.info(f"Got data: {self.got_traj_data}")
        if self.got_traj_data:
            self._data = read_csv(
                self.got_traj_data, sep=r"\s+", dtype=np.float64, comment="#"
            ).values
            names = ["ID", "FR", "X", "Y", "Z", "A", "B", "ANGLE", "COLOR"]
            self._df = pd.DataFrame(self._data, columns=names)

    def read_geo_data(self) -> Document:
        """Return xml object from geoemtry file"""
        logging.info(f"geo: {self.uploaded_traj_file}")
        if self.uploaded_geo_file:
            geo_xml = parseString(self.uploaded_geo_file.getvalue())

        elif self.selected_geo_file:
            geo_xml = parse(self.selected_geo_file)

        else:
            geo_xml = parse(self.default_geometry_file)

        return geo_xml

    def __post_init__(self) -> None:
        selection = Utilities.selected_traj_geo(self.from_examples)
        if selection:
            name_selection = selection[0]
            self.selected_traj_file = name_selection + ".txt"
            self.selected_geo_file = name_selection + ".xml"
            if name_selection not in st.session_state.example_downloaded:
                st.session_state.example_downloaded[name_selection] = True
                Utilities.download(selection[1], self.selected_traj_file)
                Utilities.download(selection[2], self.selected_geo_file)

        self.got_traj_data = self.selected_traj_file or self.uploaded_traj_file
        if self.uploaded_traj_file:
            traj_name = self.uploaded_traj_file.name.split(".txt")[0]
        if self.selected_traj_file:
            traj_name = self.selected_traj_file.split(".txt")[0]

        self.read_traj_data()
        if self.got_traj_data:
            Utilities.touch_default_geometry_file(
                self._data, st.session_state.unit, self.default_geometry_file
            )
        self.read_geo_data()
