[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/pedestriandynamics/dashboard/main/app.py)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7697604.svg)](https://doi.org/10.5281/zenodo.7697604)



# JuPedSim-Dashboard

Show statistics and make plots extracted from [jpscore](https://github.com/jupedsim/jpscore)-simulations and [experimental data](https://ped.fz-juelich.de/db/).


<img width="1043" alt="Screen Shot 2022-03-25 at 20 47 58" src="https://user-images.githubusercontent.com/5772973/160191551-4e030612-e034-4c4c-af9c-38be83036e33.png">

## Use

This app can be used online: 

https://go.fzj.de/dashboard

or locally by running 

```bash
streamlit run app.py
```

To install the requirements use:

```bash
pip install -r requirements.txt
```

## Draw geometries 

To draw geometries on trajectory-plots, try to connect all lines, such that they form a closed polygon.
This is only necessary if you want to visualize the geometry with [jpsvis](https://www.jupedsim.org/).

You can download your drawing as an xml file according to jupedsim's [format](https://www.jupedsim.org/jpscore_geometry.html)

It is possible to draw: 
- Lines 
- Rectangles

Lines and rectangle can be rotated and scaled.

To try out:
https://go.fzj.de/geometry

# Ressources
Demo: https://youtu.be/4xTqSbllCwg
Talk in the [deRSE23](https://de-rse23.sciencesconf.org/) from some scientific motivation of this project is available [here](https://zenodo.org/record/7697604).


**NOTE**: The script tries to exctract the unit of the data from the trajectory file. A good idea might be to convert the trajetories before using with this [script](https://github.com/JuPedSim/jpscore/blob/master/scripts/petrack2jpsvis.py)

or add a header with the unit information. For example a line starting with 
`#unit: cm`

## Profiles 
The density profile uses the speed of the pedestrians calculated by the Weidmann diagram.
Therefore, for jpscore-simulations, use the `optional_output`-option in the inifile:

```xml
<file location="bottleneck_traj.txt" />
    <optional_output speed="TRUE"/>
</trajectories>
```

## NT-curve 

NT-curves and flow at lines (`transitions` or [area_L](https://www.jupedsim.org/jpsreport_inifile#measurement-area))
are calculates be testing for every agent, if croses the line.
Therefore, define in the geometry transitions, such that the pedestrian fully pass them. 

    


  
