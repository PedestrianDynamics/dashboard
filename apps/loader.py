from hydralit import HydraHeadApp
#from hydralit_components import HyLoader, Loaders

class MyLoadingApp(HydraHeadApp):
    def __init__(self, title="Loader", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self, app_target):
        app_title = ""
        if hasattr(app_target, "title"):
            app_title = app_target.title

        # if app_title == "Time series":
        app_target.run()
        # else:
        #     with HyLoader("Now loading {}".format(app_title), loader_name=Loaders.standard_loaders,index=[3,0,5]):
        #         app_target.run()
