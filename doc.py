from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).parent.absolute()


def doc_plots():
    st.write(
        """
    Plot several timeseries of relevant quantities.
    Some quantities are calculated w.r.t a 2D line, which can be
    a `transition` or `area_L`.  
    See JuPedSim documentation.

    These lines are extracted from the geometry file.
    
    #### N-T
    For each line, calculate the cumulative number of pedestrians
    ($N$) passing the line at time ($T$).

    #### Flow
    For each line, calculate the flow ($J$) versus time ($T$).
    Given $N$ pedestrians have passed the line in a time duration of $T$, then the flow is calculates as:
    """
    )
    st.latex(
        r"""
    \begin{equation}
    J = \frac{N - 1}{\Delta t}.
    \end{equation}
    """
    )

    st.write(
        """
    #### Distance-Time
    Relation between time and distance to entrance.
    For each person, the Euclidean distance between the current position and the **first** selected entrance
    is calculated.
    The time to entrance is given by the time the person needs to arrive at the entrance.
    """
    )
    st.image(
        f"{ROOT_DIR}/figs/distance-time.png",
        caption="Relation between time and distance to entrance. Fig.6 in https://doi.org/10.1371/journal.pone.0177328.g006",
    )

    st.write(
        """
    #### Survival
    The time lapses $\delta$ between two consecutive agents passing a line are calculated.
    The value of $\delta$ reflects the sustained time of clogs interrupting the flow.
    This curve shows the probability distribution function
    $$P(t > δ)$$, also known as the survival function,
    which is an indicator of clogging in front of exits.
    """
    )
    st.image(
        f"{ROOT_DIR}/figs/survival_function.png",
        caption="The survival functions w.r.t door widths. Fig.7 (a) in https://doi.org/10.1016/j.physa.2021.125934",
    )


def doc_neighbors():
    st.write(
        """
    The nearest neighbors of pedestrians are retrieved using the algorithm `sklearn.neighbors.KDTree` for fast
    calculations.
    This module shows the following statistics:
    - **For a pedestrian** $i$:
       - show the time serie of the area formed by its neighbors
       - show the time serie of its contact index according to Eq. (4) in https://www.mdpi.com/2071-1050/12/22/9385

    - **For all pedestrians**:
       - show the PDF of distances at a certain frame `fr`.
       - show the PDF of distances for all frames.
       - show time serie of all distances. 
    """
    )


def doc_jam():
    st.write(
        """
    A pedestrian $i$ is defined as *congested* if its speed at time $t$
    is below a certain threshold $\hat v$: $v_i(t)<\hat v$.  
    Hence, the set of *congested* pedestrians is defined as
    """
    )
    st.latex(
        r"""
    \begin{equation}
    C(t) = \{i |\; v_i(t) < \hat v \}.
    \end{equation}
    """
    )
    st.info(
        """
    Therefore, we define **jam** as a state, where at least
    $\hat n$ pedestrians are *congested* for a certain amount of time $\hat t$.
    """
    )

    st.write(
        """
    To quantify the characteristics of a jammed state we define the following
    quantities:
    
    #### Maximal waiting time

    The maximal waiting time of congested pedestrians is defined
    as the longest time spent in jam
    """
    )

    st.latex(
        r"""
    \begin{equation}
    T_{w} = \max_i  \{\Delta t = t_{i,2}  - t_{i,1} |\; \Delta t > \hat t,  v(t_{i_1}) < \hat v\; {\rm and}\; v(t_{i_2}) > \hat v \}, 
    \end{equation}
    """
    )
    st.write("""where $\hat t$ is the minimal jam duration.""")
    st.write(
        """
    ####  Lifetime of jam
    Considering a minimal number of pedestrians in jam $\hat n$, and the set of congested pedestrians Eq. (1), the longest time period of a jam is
    """
    )
    st.latex(
        r"""
        \begin{equation}
        T_l = \max_{\Delta t}\{\Delta t = t_{i,2}  - t_{i,1} |\; C(t_{i_1}) > \hat n\; {\rm and}\; C(t_{i_2}) < \hat n \}.
        \end{equation}
        """
    )
    st.write(
        """
    ####  Size of jam
    The number of pedestrians in jam during its lifetime is the mean value of the number of congested pedestrians:    
    """
    )
    st.latex(
        r"""
        \begin{equation}
        \mu \{|C(t) |\;  t \in I\},
        \end{equation}
        """
    )
    st.write(
        """where $I$ is the time interval corresponding
    to the lifetime of time. See Eq. (3)."""
    )

    st.write(
        """
    #### Summary of jam parameters
    | Variable    | Notation |
    |--------------|-----------|
    |**Min Jam Speed**  | $\hat v$|
    |**Min Jam Duration** | $\hat t$|
    |**Min Agents in Jam** | $\hat n$|
    """
    )


def doc_speed():
    st.write(
        """
    #### Speed
     The speed can be calculated *from simulation*: in this case
     use in the inifile the option: `<optional_output   speed=\"TRUE\">`.

     Alternatively, the speed can be calculated *from trajectory*
     according to the forward-formula:
     """
    )
    st.latex(
        r"""
    \begin{equation}
    v_i(f) = \frac{x_i(f+df) - x_i(f))}{df},
    \end{equation}
    """
    )
    st.write(
        r"""with $df$ a constant and $v_i(f)$ the speed of pedestrian $i$ at frame $f$."""
    )


def doc_timeseries():
    st.write(
        """
    Time series of the density and the speed are calculated within a measurement rectangle with side lengths $dx$ and $dy$.

    Density and speed are calculated as defined in `Profiles`, whereas flow is defined as:
    """
    )

    st.latex(
        r"""
    J = \rho\cdot v / l,
"""
    )
    st.write(
        """
        where $l$ is a constant between $dx$ and $dy$.
        
    Depending on the frames per seconds of the trajectories, it might be better to increase the sampling rate
    (`sample`) to speed up rendering the plots.
    """
    )


def doc_profile():
    st.write(
        """
    The density and speed profiles show averaged values over time and over space.

    A grid of square cells $c$ with a given size (can be defined by the slider `Grid size`) is created.
    The values of the density and speed are then averaged over the cells over time.

    Different methods can be used: `Classical`, `Gaussian` and `Weidmann`

    #### Weidmann

    Given the Weidmann-formula **[Weidmann1992 Eq. (15)]**:
    """
    )
    st.latex(
        r"""
    \begin{equation}
    v_i = v^0 \Big(1 - \exp\big(\gamma (\frac{1}{\rho_i} - \frac{1}{\rho_{\max}}) \big)  \Big).
    \end{equation}
    """
    )
    st.text("Eq. (1) can be transformed in ")
    st.latex(
        r"""
    \begin{equation}
    \rho_i = \Big(-\frac{1}{\gamma} \log(1 - \frac{v_i}{v^0})+ \frac{1}{\rho_{\max}}\Big)^{-1},
    \end{equation}
    """
    )
    st.write("""where""")
    st.latex(
        r"""\gamma = 1.913\, m^{-2},\; \rho_{\max} = 5.4\, m^{-2}\; \;{\rm and}\; v^0 = 1.34\, m/s."""
    )
    st.write(
        "Based on the speed, from simulation or trajectory, and using Eq. (2) we can calculate the density $\\rho_i$ and hence,"
    )
    st.latex(
        r"""
    \rho_c = \frac{1}{T}\sum_{t=0}^T S_c,
    """
    )
    st.write("where $S_c$ is the sum of $\\rho_i$ in $c$ and $T$ the evcuation time.")
    st.write("""#### Classical  """)
    st.latex(r"""\rho_c = \frac{1}{T}\sum_{t=0}^T \frac{N_c}{A_c},""")
    st.write("where $A_c$  the area of cell $c$ and $N_c$ the number of agents in $c$.")
    st.write(
        """#### Gaussian
For every pedestrian $i$ the density field over the whole geometry is used. The local density $\\rho$ in the system can be defined as 
        """
    )
    st.latex(
        r"""
    \begin{equation}
    \rho(\mathbf{r;\mathbf{X}}) = \sum_{i=1}^{N} \delta(\mathbf{r}_i - \mathbf{r}), \quad \rho(\mathbf{r}) = \langle \rho(\mathbf{r}; \mathbf{X}) \rangle,
    \end{equation}
    """
    )
    st.write(
        r"""
    where $\textbf{r}$ is the position and $\textbf{X}$ marks a configuration and $\delta(x)$ is approximated by a Gaussian
    """
    )

    st.latex(
        r"""
    \begin{equation}
    \delta(x) =\frac{1}{\sqrt{\pi} a } \exp[-x^2/a^2].
    \end{equation}
    """
    )
    st.write("Finally, the average of the density per cell is")
    st.latex(r"""\rho_c = \frac{1}{T}\sum_{t=0}^T \rho(\mathbf{r;\mathbf{X}}) ,""")
    st.write(
        """
The speed is calculated from $\\rho_i$ by Eq. (1).
    """
    )
    st.markdown("--------")
    st.write("#### References:")
    st.code(
        "Weidmann1992: U. Weidmann, Transporttechnik der Fussgänger: Transporttechnische Eigenschaften des Fussgängerverkehrs, Literaturauswertung, 1992"
    )


def docs():
    st.write(
        """
        ### :information_source: About this dashboard
        
        This is an interactive visual tool for explorative analysis and inspection of pedestrian dynamics based on trajectories.
        The input data are

        - Trajectories of pedestrians. The file can be a [jpscore](https://github.com/jupedsim/jpscore)-simulation or [experimental data](https://ped.fz-juelich.de/db/).
        - A geometry file.

        ### :point_left: Examples

        To use this dashboard, you can either
        1. Explore one of the provided examples.
        2. Or upload your own trajectory and geometry files.

        ### :bar_chart: Measurements

        The dashboard is organized in decoupled tabs that can be used independently from each others.

        - **Data summary**: Gives a brief overview of the imported trajectory file.
        - **Trajectories**: Plot trajectories and measurement lines. Optionally, a single pedestrian can be plotted as well.
        - **Jam**: Some relevant Jam-quantities are calculated and plotted, e.g.:
          - Jam waiting time [[Sonntag2020](https://fz-juelich.sciebo.de/s/rqcrLVT5v7R9icI)].
          - Jam life span [[Sonntag2020](https://fz-juelich.sciebo.de/s/rqcrLVT5v7R9icI)].
        - **Statistics**: Some common quantities are calculated at **measurement lines** and plotted. These are:
          - N-T curves.
          - T-D (time-distance) curves [[Adrian2020](https://collective-dynamics.eu/index.php/cod/article/view/A50)].
          - Flow vs time.
          - Survival function  [[Xu2021](https://www.sciencedirect.com/science/article/abs/pii/S0968090X21004502)].
          - Discharge function: The change of the number of pedestrians inside a room w.r.t to time.
        - **Profiles**: Density and speed heatmaps [[Zhang2012](https://arxiv.org/abs/1112.5299)]. Different methods for density and speed calculation. See documentation inside the tab.
        - **Time-series**:  Density, speed and flow time series. Here, you can draw measurement areas and calculate these quantities inside.
        - **RSET** heatmaps [[Schroeder2017](http://elpub.bib.uni-wuppertal.de/servlets/DerivateServlet/Derivate-7013/dd1611.pdf)]
        """,
        unsafe_allow_html=True,
    )


def doc_RSET():
    st.write(
        """
    RSET maps are defined in Schroeder2017 [1] are a spatial representation of the required safe egress time.
    In a regular grid, the time for which, the cell was last occupied by a pedestrian is calculated.

    These maps give insight about the location of potential jam areas.
    More importantly they  highlight the used exits in the scenario.

    [1]: Multivariate methods for life safety analysis in case of fire
    """
    )
