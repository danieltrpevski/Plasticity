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
	POINT_PROCESS adaptive2_hetero_bcm_inhexp2syn
	RANGE tau1, tau2, e, i, tau
	NONSPECIFIC_CURRENT i
	RANGE g, gmax
	RANGE delta_t, tlast
	RANGE weight, calcium_max, learning_rate, n, w0, active_syn_flag, last_wuf
	RANGE learning_rate_theta, theta, theta_min, kernel, sf
	POINTER weight_update_flag
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
	calcium_max = 0 (mM)
	active_syn_flag = 0
	last_wuf = 0
	learning_rate = 0
	learning_rate_theta = 0
	n = 10
	w0 = 0.0
	tau = 1000 (ms)
	tlast = 0
	delta_t = 0
	gmax = 0.1 (uS)
	a = 0.3
	sf = 10000
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
	kernel_theta
	kernel_theta_min
	weight
	weight_update_flag
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
	caint = 0
	
	tlast = 0
	delta_t = 0
	
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g*gmax*(v - e)
	
	delta_t = t - tlast
	tlast = t
	
	if (weight_update_flag == 0) {
	        if (last_wuf == 1) {
	             reset_asf()
	             reset_max()
	        }
        	calcium_max = max(caint, calcium_max)
        } else {
	  LOCAL s1, s2
	  s1 = sigmoidal(sf*calcium_max, theta_min, n)
	  s2 = sigmoidal(sf*calcium_max, theta, n)
	  
          if (active_syn_flag == 1) {	
          	kernel = s1 * (2*s2 - 1) 
	  } else {
	  	kernel = -1 * s1 * (2*s2 - 1) 
	  }
	  
	  weight = weight + learning_rate * kernel * delta_t
	  theta = theta + learning_rate_theta * (sf*calcium_max - theta) * delta_t
	  
	  if (weight < 0) {
		weight = 0
	  }
	  
	  if (theta < theta_min) {
	  	theta = theta_min
          }
	  
	}
	
	last_wuf = weight_update_flag
}

DERIVATIVE state {
	A' = -A/tau1
	B' = -B/tau2
	caint' = (cali-caint)/tau
}

NET_RECEIVE(dummy (uS)) {
	A = A + weight*factor
	B = B + weight*factor
	
	active_syn_flag = 1
}

FUNCTION sigmoidal(x, x_offset, s) {
    sigmoidal = 1/(1+exp(-s *(x - x_offset)))
}

PROCEDURE reset_max() {
	calcium_max = 0
}

PROCEDURE reset_asf() {
	active_syn_flag = 0
}
        
FUNCTION max(current, maximum) {
   if (current>maximum) { 
      max = current
   } else {
      max = maximum
   }
}

