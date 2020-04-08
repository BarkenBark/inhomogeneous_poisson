# PSL
import argparse

# External
import numpy as np
from numpy import pi as pi, sqrt as sqrt, exp as exp
from scipy.special import erf
#from pynverse import inversefunc
import seaborn as sns


# TO-DO:
######################


# EXAMPLE PROCESSES
###############################
# A nonhomogenous Poisson process is defined by a intensity function lambda(t) valid on an (possible infitine) interval
# For some simulation methods, it is useful to provide the cumulative internsity function Lamba(t) which is the primitive function to lambda(t)
# For some simulation methods, it is useful to provide the inverse cumulative intensity function Lambda_inv(x) which is the inverse of the cumulative intensiy function

class IHPP():

    # Must be overridden
    def intensity(self, t):
        raise NotImplementedError

    # May be overridden
    def cumulative_intensity(self, t):
        raise NotImplementedError

    # May be overridden
    def inverse_cumulative_intensity(self, i):
        raise NotImplementedError

class LinearRateIHPP(IHPP):

    def __init__(self, a, b):
        assert a != 0
        self.a = a
        self.b = b
        assert approx_equals(self.inverse_cumulative_intensity(self.cumulative_intensity(5)), 5) 

    def intensity(self, t):
        """ Intensity function

        Parameters
        ----------
        t : float in range [0,24)
            Time of day in hours

        Returns
        ------------
        lambda : float in R+
            Instantaenous event rate in events/hour
        """
        return self.a*t + self.b

    def cumulative_intensity(self, t):
        return self.a/2*t**2 + self.b*t

    def inverse_cumulative_intensity(self, i):
        return (np.sqrt(2*self.a*i + self.b**2) - self.b) / self.a

class TrafficIHPP():
    """Collection of functions defining an inhomogenous poisson process modeling traffic
    """
    
    @staticmethod
    def get_params():
        A = 500
        a = 1.6
        b1 = 8
        b2 = 17
        C = 60
        return A, a, b1, b2, C

    @staticmethod
    def intensity(t):
        """Returns the instantaneous rate of occurences for a traffic situation

        Event = Car passes by on a street.
        This model assumes that we have a rush hour at around 8 in the morning and around 5 in the evening
        
        Parameters
        ----------
        t : float in range [0,24)
            Time of day in hours

        Returns
        ------------
        lambda : float in R+
            Instantaenous event rate in events/hour
        """
        A, a, b1, b2, C = TrafficIHPP.get_params()
        return A*np.exp(-a*(t-b1)**2/2) + A*np.exp(-a*(t-b2)**2/2) + C

    @staticmethod
    def cumulative_intensity(t):
        A, a, b1, b2, C = TrafficIHPP.get_params()
        return sqrt(pi/2/a) * A * ( erf(sqrt(a/2)*(t-b1)) + erf(sqrt(a/2)*(t-b2)) ) + C*t

    @staticmethod
    def inverse_cumulative_intensity(t):
        raise NotImplementedError








# SIMULATION 
###############################

def inhomogenous_poisson_thinning(intensity_fun, t_start, t_end, lambda_bar=None, n_trials=1):
    """Simulate an inhomogenous poisson process using the Thinning method
    
    Based on the Thinning Method outlined by SecretAgentMan in https://stats.stackexchange.com/questions/369288/nonhomogeneous-poisson-process-simulation

    NOTE: Inefficient when intensity fluctuation in time is large because big lambda_bar gives a high rejection probability. 
    Possible workaround is to break interval [0,T] into small intervals and pick a lambda_bar for each interval.

    Parameters
    ----------
    intensity_fun : function mapping float in R to float in R+
        Function returning instantaneous event rate lambda at time t
    t_start : float
        Start sampling at t_start
    t_end : float
        Discontinue sampling after t_end time units
    lambda_bar : float
        Parameter greater than or equal to the largest value intensity_fun can take on interval [t_start, t_end]
    n_trials : int, optional
        Number of trials

    Returns 
    -----------------------
    event_times : list of np.ndarray, len=n_trials OR np.ndarray
        List of event time arrays where the shape of each ndarray is the number of events on the time interval for each trial.
        If n_trials=1, only return ndarray with event times for the single trial
    """

    assert t_end > t_start
    assert n_trials >= 1

    if lambda_bar is None:
        raise NotImplementedError("Automatic inference of lambda_bar not yet implemented. Please provide an appropriate value of lambda_bar manually.")
    
    event_times = []
    for _ in range(n_trials):
        t = t_start
        event_times_trial = []
        while True:
            U1 = np.random.rand()
            t = t - np.log(U1/lambda_bar)
            if t > t_end:
                break
            U2 = np.random.rand()
            acceptance_probability = intensity_fun(t) / lambda_bar
            assert acceptance_probability <= 1, "Acceptance probability greater than 1, too low lambda_bar value has been provided."
            if U2 <= acceptance_probability:
                event_times_trial.append(t)
        event_times.append(np.array(event_times_trial))

    if n_trials > 1:
        return event_times
    else:
        return event_times[0]

def inhomogenous_poisson_ntt(inverse_cumulative_intensity_fun, t_start, t_end, n_trials=1):
    """Simulate an inhomogenous poisson process using the Nonlinear Time Transformation method
    
    Based on the Nonlinear Time Transformation method outlined by SecretAgentMan in https://stats.stackexchange.com/questions/368968/non-homogenous-poisson-process-with-simple-rates/369093#369093

    Parameters
    ----------
    inverse_cumulative_intensity_fun : 
    t_start : float
        Start sampling at t_start
    t_end : float
        Discontinue sampling after t_end time units
    n_trials : int, optional
        Number of trials

    Returns 
    -----------------------
    event_times : list of np.ndarray, len=n_trials
        List of event time arrays where the shape of each ndarray is the number of events on the time interval for each trial
    """
    assert t_end > t_start
    assert n_trials >= 1

    event_times = []
    for _ in range(n_trials):
        t = t_start
        event_times_trial = []
        while True:
            arrival_time_base = np.random.exponential(scale=1)
            arrival_time = inverse_cumulative_intensity_fun(arrival_time_base)
            t += arrival_time
            if t > t_end:
                break
            event_times_trial.append(t)
        event_times.append(np.array(event_times_trial))

    if n_trials > 1:
        return event_times
    else:
        return event_times[0]

def inhomogenous_poisson_cinlar(cumulative_intensity_fun, t_end, t_start=0, n_trials=1):
    """Simulate an inhomogenous poisson process using a method proposed by Cinlar

    Based on Cinlar method outlined by Freakonometrics in https://freakonometrics.hypotheses.org/724

    Parameters
    ----------
    cumulative_intensity_fun : 
    t_start : float
        Start sampling at t_start
    t_end : float
        Discontinue sampling after t_end time units
    n_trials : int, optional
        Number of trials

    Returns
    -----------------------
    event_times : list of np.ndarray, len=n_trials
        List of event time arrays where the shape of each ndarray is the number of events on the time interval for each trial    
    """
    if t_start != 0: raise NotImplementedError("Can't handle non-zero start times atm.")
    assert t_end > t_start
    assert n_trials >= 1

    #TODO: Replace with more efficient function
    def get_infimum(f, y, x_interval):
        """Returns the infinimum of the set of values x which satisfy f(x)>y
            NOTE: Only consideres values on the defined interval x_interval
        """
        y_interval = f(x_interval)
        canidates = x_interval[y_interval > y]
        if len(canidates)==0:
            return None
        else:
            return min(x_interval[y_interval > y])

    infimum_interval_res = 1000
    event_times = []
    for _ in range(n_trials):
        s = 0 # Could I make code more logical by initializing s such that t only takes on values greater than t_start?
        event_times_trial = []
        while True:
            u = np.random.rand()
            s = s - np.log(u)
            t = get_infimum(cumulative_intensity_fun, s, np.linspace(0, t_end, num=infimum_interval_res, endpoint=True)) # Can I make more efficient by starting at current t?
            if t is None: # Meaning no t > t_end was found in infimum search
                break
            event_times_trial.append(t)
        event_times.append(np.array(event_times_trial))

    if n_trials > 1:
        return event_times
    else:
        return event_times[0]

def inhomogenous_poisson_exp(cumulative_intensity_fun, inverse_cumulative_intensity_fun, t_start, t_end, n_trials=1):
    """Simulate an inhomogenous poisson process using a method sampling event times directly from exponential distribution

    Based on method outlined by Freakonometrics in https://freakonometrics.hypotheses.org/724

    NOTE: If inverse_cumulative_intensity_fun is not available, you can approximate inverse_cdf_arrival_time below using the Bisection Algorithm to solve for x in the equation x = f_inv(y) on the interval [t_start, t_end]

    Parameters
    ----------
    cumulative_intensity_fun : 
    inverse_cumulative_intensity_fun : 
    t_start : float
        Start sampling at t_start
    t_end : float
        Discontinue sampling after t_end time units
    n_trials : int, optional
        Number of trials

    Returns
    -----------------------
    event_times : list of np.ndarray, len=n_trials
        List of event time arrays where the shape of each ndarray is the number of events on the time interval for each trial    
    """
    assert t_end > t_start
    assert n_trials >= 1

    # Aliases for readability
    Lambda = cumulative_intensity_fun
    Lambda_inv = inverse_cumulative_intensity_fun

    event_times = []
    for _ in range(n_trials):
        t = t_start
        event_times_trial = []
        while True:
            inverse_cdf_arrival_time = lambda y: Lambda_inv(Lambda(t)-np.log(1-y))-t
            t = t + inverse_transform_sampling(inverse_cdf_arrival_time)
            if t > t_end:
                break
            event_times_trial.append(t)
        event_times.append(np.array(event_times_trial))
        
    if n_trials > 1:
        return event_times
    else:
        return event_times[0]
            

# VALIDATION
############################
def index_of_dispersion(event_times):
    """Get the Index of Dispersion for event counts
    
    Parameters
    ----------
    event_times : list of np.ndarray
        event_times[i] contains an ndarray of event times for the i:th trial
    """
    raise NotImplementedError



# PLOTTING
##############################

def plot_rug_dist(ax, event_times, unit="h"):
    """Plots a kernel estimate over a rubplot for a single realization of a event process
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot into
    event_times : np.ndarray, shape=[n_events]
        Event times of a single process realization
    unit : str
    """
    sns.distplot(event_times, rug=True, ax=ax)
    ax.set_xlabel(f"Event time ({unit})")
    ax.set_ylabel("Density")

def plot_timelines(ax, event_times, unit="h"):
    """Plots timelines for each of n_trials process realizations next to each other
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot into
    event_times : list of np.ndarray, len=n_trials
        event_times[i] contains an ndarray of event times for the i:th trial
    unit : str
    """
    n_timelines = len(event_times)
    timeline_heights = np.linspace(0, 1, num=n_timelines+2)[1:-1]
    for i in range(n_timelines):
        ax.scatter(event_times[i], timeline_heights[i]*np.ones(len(event_times[i])), marker='|')
    ax.set_ylim(0,1)
    ax.set_yticks(timeline_heights, minor=False)
    ax.set_yticklabels(np.arange(n_timelines)+1)
    ax.yaxis.grid(True, which='major')
    ax.set_ylabel('Timeline #')
    ax.set_xlabel(f"Time ({unit})")

# UTILITY
#########################

def event_counts(event_times, t):
    """Return the number of events which have occured at time t given a realization of an event process
    
    Parameters
    ----------
    event_times : np.ndarray
    t : float
    
    Returns
    -------
    n_events : int
    """
    return (event_times <= t).sum()

def get_event_counts_fun(event_times):
    """Returns a function which returns the event count at time t given historical event times
    
    Parameters
    ----------
    event_times : np.ndarray
        Historical event times of process realization for which to generate the count function
    
    Returns
    ---------------
    count_fun : function mapping float to int
        Function which returns how many events have occured at time t
        NOTE: Not reliable for t > max(event_times)
    """
    def event_counts_fun(t):
        return (event_times <= t).sum()

    return event_counts_fun

def inverse_transform_sampling(inverse_cdf, n_samples=1):
    """Perform inverse transform sampling to obtain samples of a continious random variable distributed with pdf f(x) by using the inverse of the cdf F_inv(y)
    
    Parameters
    ----------
    inverse_cdf : function
        Inverse cumulative density function, taking a probability p and returning the value x of the random variable X such that P(X<=x)=p
    n_samples : int
    
    Returns
    -------
    x : np.ndarray, shape=[n_samples]
    """
    u = np.random.rand(n_samples)
    return inverse_cdf(u)

def inverse_transform_sampling_discrete(pmf, n_samples):
    """Perform inverse transform sampling to obtain samples of a discrete random variable distributed with pmf p(x)
    
    Parameters
    ----------
    pmf : function
        Probability mass function
    n_samples : int
    
    Returns
    -------
    x : np.ndarray, shape=[n_samples]
    """
    raise NotImplementedError

def get_inverse_fun_bisection(f, a0, b0, bisection_iter=20):
    """Return a function handle to compute the inverse function of f
    
    Parameters
    ----------
    f : function mapping float to float
        Should be monotonically increasing in order for the inverse to be valid
    a : float
        left bound for bisection method
    b : float
        right bound for bisection method
    bisection_iter : int, optional
        Number of bisection methood iterations to perform in a function call, by default 20
    """
    resolution = 10000
    vals = f(np.linspace(a0, b0, resolution))
    assert all(np.diff(vals)>=0), "Provided function f is not monotonically increasing on the interval (a0, b0)"

    return lambda y: inverse_fun_bisection(f, y, a0, b0, bisection_iter=bisection_iter)

def inverse_fun_bisection(f, y, a0, b0, bisection_iter=20):
    """ Evaluate the inverse function of f at y by using the bisection method
    
    Parameters
    ----------
    f : function mapping float to float
        Should be monotonically increasing in order for the inverse to be valid
    y : float
        Value for which to evaluate f_inv
    a : float
        left bound for bisection method
    b : float
        right bound for bisection method
    bisection_iter : int, optional
        Number of bisection methood iterations to perform in a function call, by default 20
    """
    a = a0
    b = b0
    for _ in range(bisection_iter):
        c = (a+b)/2
        if f(c)<=y:
            a = c
        else:
            b = c
    return (a+b)/2 

def approx_equals(tol=10e-6, *args):
    """Return true if all real-valued values in args are approximately equal

    Parameters
    ------------------
    tol : float
        Absolute error of the difference with greatest absolute error should be less than tol
    args:
        Values to compare
    """
    max_error = 0
    for i in range(len(args)):
        for i in range(i+1, len(args)):
            error = abs(args[i]-args[j])
            max_error = error if error > max_error else max_error
    return max_error <= tol




def main():
    raise NotImplementedError

if __name__=="__main__":
    main()