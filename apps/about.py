import sys

sys.path.append('../')
from hydralit import HydraHeadApp

#add an import to Hydralit
import doc


class AboutClass(HydraHeadApp):

    def run(self):
        doc.docs()
