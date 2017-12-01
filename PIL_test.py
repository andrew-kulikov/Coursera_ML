import numpy as np
import pylab

x = (np.arange(100) - 50) / 10
pylab.plot(x, np.sin(x))
pylab.plot(x, np.cos(x))
pylab.show()