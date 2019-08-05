"""
主要是用来分析实验过程数据
"""
import numpy as np
import matplotlib.pyplot as plt

def training_parser(analysis_matrix: np.ndarray, save_path: str):
    """
    :param analysis_matrix: a matrix.
    i.e. A. B. C.
    loss 1  1  1
    ppl  2  2  2

    If you want to show the data, choose a number to show them in a subfigure.
    If you don't, please enter 0.

    :param save_path: save path.
    :return:
    """


    fig = plt.figure()
    rows = analysis_matrix.max()
