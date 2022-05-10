import sys

sys.path.append('../')
#add an import to Hydralit
import doc
from hydralit import HydraHeadApp

class AboutClass(HydraHeadApp):

    def run(self):
        doc.docs()
