if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    xarr = np.linspace(0, 100, 100)
    yarr = np.random.normal(0, 1, 100)
    plt.plot(xarr, yarr)
