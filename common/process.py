import os
import pandas as pd
class Process:
    """
    数据处理的过程，目前只对于csv格式的数据进行处理
    """
    def __init__(self,path):
        self.file = path

    def load_file(self):
        if os.path.exists(os.path.join('dataset',self.file)):
            data = pd.read_csv(os.path.join('dataset',self.file))

