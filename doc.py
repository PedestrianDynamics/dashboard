import streamlit as st


def doc_plots():
    st.write("""
    Plot several timeseries of relevant quantities.
    Some quantities are calculated w.r.t a 2D line, which can be
    a `transition` or `area_L`.  
    See JuPedSim documentation.

    These lines are extracted from the geometry file.
    
    ### N-T
    For each line, calculate the cumulative number of pedestrians
    ($N$) passing the line at time ($T$).

    ### Flow
    For each line, calculate the flow ($J$) versus time ($T$).
    Given $N$ pedestrians have passed the line in a time duration of $T$, then the flow is calculates as:
    """)
    st.latex(
        r"""
    \begin{equation}
    J = \frac{N - 1}{\Delta t}.
    \end{equation}
    """
    )

    st.write("""
    ### Occupation
    This `discharge curve` shows the number of pedestrians inside the scenario versus time.

    ### Survival
    The time lapses $\delta$ between two consecutive agents passing a line are calculated.
    The value of $\delta$ reflects the sustained time of clogs interrupting the flow.
    This curve shows the probability distribution function
    $$P(t > δ)$$, also known as the survival function,
    which is an indicator of clogging in front of exits.
    """)
    st.image("./figs/survival_function.png", caption="The survival functions w.r.t door widths. See: Fig.7 (a) in https://doi.org/10.1016/j.physa.2021.125934")


def doc_jam():
    st.write("""
    Work in progress
    """
    )


def doc_speed():
    st.write("""
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


def doc_profile():
    st.write("""
    The density and speed profiles show averaged values over time and over space.

    A grid of square cells $c$ with a given size (can be defined by the slider `Grid size`) is created.
    The values of the density and speed are then averaged over the cells over time.

    Different methods can be used: `Classical`, `Gaussian` and `Weidmann`

    **1. Weidmann**

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
    st.write("Based on the speed, from simulation or trajectory, and using Eq. (2) we can calculate the density $\\rho_i$ and hence,")
    st.latex(
        r"""
    \rho_c = \frac{1}{T}\sum_{t=0}^T S_c,
    """
    )
    st.write("where $S_c$ is the sum of $\\rho_i$ in $c$ and $T$ the evcuation time.")
    st.write("""**2. Classical**  """)
    st.latex(r"""\rho_c = \frac{1}{T}\sum_{t=0}^T \frac{N_c}{A_c},""")
    st.write("where $A_c$  the area of cell $c$ and $N_c$ the number of agents in $c$.")
    st.write(
    """**3. Gaussian**  
    for every pedestrian $i$ a Gaussian distribution is calculated, then
    """
    )
    st.latex(r"""\rho_c = \frac{1}{T}\sum_{t=0}^T G_c,""")
    st.write("""where $G_c$ the sum of all Gaussians.
    The speed is calculated from $\\rho_i$ by Eq. (1).
    """)
    st.markdown("--------")
    st.write("#### References:")
    st.code(
        "Weidmann1992: U. Weidmann, Transporttechnik der Fussgänger: Transporttechnische Eigenschaften des Fussgängerverkehrs, Literaturauswertung, 1992"
    )


    
def docs():
    st.write(
        """
    This app performs some basic measurements on data simulated by jpscore.

     ### Flow
     Flow and NT-curves are calculated at transitions and measurement lines.

     Measurement lines `area_L` can be added as defined in
     https://www.jupedsim.org/jpsreport_inifile#measurement-area
        """
    )
    
        
     
