import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from main import plot_timelines
from main import inhomogenous_poisson_thinning, inhomogenous_poisson_exp, inhomogenous_poisson_cinlar, inhomogenous_poisson_ntt
from main import LinearRateIHPP

event_times = []
for i in range(5):
    cutoff = np.random.randint(5)
    this_event_times = np.random.rand(20)[0:-(cutoff+1)]
    event_times.append(this_event_times)

t_start = 0
t_end = 10
base_rate = 10
linear_ihpp = LinearRateIHPP(base_rate, 0)
intensity_fun = linear_ihpp.intensity
cumulative_intensity_fun = linear_ihpp.cumulative_intensity
inverse_cumulative_intensity_fun = linear_ihpp.inverse_cumulative_intensity
# intensity_fun = lambda t: base_rate*t
# cumulative_intensity_fun = lambda t: base_rate/2*t**2
# inverse_cumulative_intensity_fun = lambda y: np.sqrt(2/base_rate*y)
assert np.abs(inverse_cumulative_intensity_fun(cumulative_intensity_fun(5))-5) < 10e-6
#event_times = inhomogenous_poisson_thinning(intensity_fun, t_start, t_end, lambda_bar=base_rate*t_end, n_trials=5) # Doesn't work. When it does, gives entirely different result than _exp-method
#event_times = inhomogenous_poisson_exp(cumulative_intensity_fun, inverse_cumulative_intensity_fun, t_start, t_end, n_trials=10)
event_times = inhomogenous_poisson_cinlar(cumulative_intensity_fun, t_end, t_start=t_start, n_trials=1000) # Doesn't work
#event_times = inhomogenous_poisson_ntt(inverse_cumulative_intensity_fun, t_start, t_end, n_trials=10) # Works, but yields entirely different results than _exp-method

a = 4
b = 5
mean_counts = 0
for i in range(len(event_times)):
    counts = ((event_times[i]>a)&(event_times[i]<b)).sum()
    mean_counts = (mean_counts*i + counts) / (i+1)
print(mean_counts)
print('Expected mean counts: ', cumulative_intensity_fun(b)-cumulative_intensity_fun(a))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plot_timelines(ax, event_times[0:10])
plt.show()