from hydralit import HydraHeadApp

# from hydralit_components import HyLoader, Loaders


class MyLoadingApp(HydraHeadApp):
    def __init__(self, title="Loader", **kwargs):
        self.__dict__.update(kwargs)
        self.title = title

    def run(self, app_target):
        # app_title = ""
        # if hasattr(app_target, "title"):
        #     app_title = app_target.title

        # if app_title == "Time series":
        app_target.run()
