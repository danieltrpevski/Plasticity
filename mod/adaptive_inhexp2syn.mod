COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 is very small compared to tau1, this is an alphasynapse with time constant tau2.
If tau1/tau2 is very small, this is single exponential decay with time constant tau2.

The factor is evaluated in the initial block 
such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
	POINT_PROCESS adaptive_inhexp2syn
	RANGE tau1, tau2, e, i, tau
	NONSPECIFIC_CURRENT i
	RANGE g, gmax
	RANGE weight, calcium, learning_rate, theta_min, theta, theta_max, kernel, n, w0, sf, caint0
	RANGE delta_t, tlast
	USEION ca_nmda READ ca_nmdai VALENCE 2	
	USEION cal READ cali VALENCE 2
	USEION cat READ cati VALENCE 2
	USEION ca READ cai VALENCE 2
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau1 = 0.1 (ms) <1e-9,1e9>
	tau2 = 10 (ms) <1e-9,1e9>
	e=-60	(mV)
	theta = 0.0 (mM)
	theta_min = 0.0 (mM)
	theta_max = 0.0 (mM)
	calcium = 0 (mM)
	learning_rate = 0
	n = 10
	w0 = 0.0
	tau = 1000 (ms)
	gmax = 0.1 (uS)
	sf = 1000
	tlast = 0
	delta_t = 0
	caint0 = 0.0

}

ASSIGNED {
	v (mV)
	i (nA)
	g (uS)
	factor
	cali
	cati
	cai
	ca_nmdai
	kernel
	weight
}

STATE {
	A (uS)
	B (uS)
	
	caint
}

INITIAL {
	LOCAL tp
	if (tau1/tau2 > 0.9999) {
		tau1 = 0.9999*tau2
	}
	if (tau1/tau2 < 1e-9) {
		tau1 = tau2*1e-9
	}
	A = 0
	B = 0
	tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
	factor = -exp(-tp/tau1) + exp(-tp/tau2)
	factor = 1/factor
	
	weight = w0
	caint = caint0
	
	tlast = 0
	delta_t = 0
	
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g*gmax*(v - e)
	
	delta_t = t - tlast
	tlast = t
	
	calcium = caint*sf
	kernel = sigmoidal(calcium, theta_min, n) * (2*sigmoidal(calcium, theta, n) - 1) : * (1 - sigmoidal(calcium, theta_max, n))
	if (kernel < 0) {
		weight = weight + learning_rate * kernel * delta_t
		if (weight < 0) {
		    weight = 0
		}
	} else {
		weight = weight + learning_rate * kernel * delta_t
	}	
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
	caint' = (cali-caint)/tau
}

NET_RECEIVE(dummy (uS)) {
	A = A + weight*factor
	B = B + weight*factor
}

FUNCTION sigmoidal(x, x_offset, s) {
    sigmoidal = 1/(1+exp(-s *(x - x_offset)))
}
