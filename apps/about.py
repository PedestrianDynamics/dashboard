import sys
from hydralit import HydraHeadApp
import doc

sys.path.append("../")


class AboutClass(HydraHeadApp):
    def run(self):
        doc.docs()
