[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/chraibi/jupedsim-dashboard/main/app.py)
[![Heroku](http://heroku-shields.herokuapp.com/jupedsim-dashboard)](https://jupedsim-dashboard.herokuapp.com/)


# JuPedSim-Dashboard

Show statistics and make plots extracted from [jpscore](https://github.com/jupedsim/jpscore)-simulations.
<img width="1159" alt="Screen Shot 2022-03-25 at 15 00 23" src="https://user-images.githubusercontent.com/5772973/160135275-35830dd6-c7c8-4522-be65-8ba5befad06d.png">



## Profiles 
The density profile uses the speed of the pedestrians the the Weidmann diagram.

Therefore, use the `optional_output`-option in the inifile:

```xml
<file location="bottleneck_traj.txt" />
    <optional_output speed="TRUE"/>
</trajectories>
```

## NT-curve 

Define in the geometry transitions, such that the pedestrian fully pass them. 
