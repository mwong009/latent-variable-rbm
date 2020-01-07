########################################
#
# @file 01logit.py
# @author: Michel Bierlaire, EPFL
# @date: Wed Dec 21 13:23:27 2011
#
#######################################

from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *

import numpy as np

#Parameters to be estimated
# Arguments:
#   - 1  Name for report; Typically, the same as the variable.
#   - 2  Starting value.
#   - 3  Lower bound.
#   - 4  Upper bound.
#   - 5  0: estimate the parameter, 1: keep it fixed.
#
asc_aval   = Beta('asc_aval  ',0,-10,10,0,'guarantees')
asc_deco   = Beta('asc_deco  ',0,-10,10,0,'short-term dp.')
asc_deme   = Beta('asc_deme  ',0,-10,10,0,'mid-term dp.')
asc_dela   = Beta('asc_dela  ',0,-10,10,0,'long-term dp.')
asc_fond   = Beta('asc_fond  ',0,-10,10,0,'funds')
asc_hip    = Beta('asc_hip   ',0,-10,10,0,'mortgage')
asc_plan   = Beta('asc_plan  ',0,-10,10,0,'pensions')
asc_pres   = Beta('asc_pres  ',0,-10,10,0,'loans')
asc_reca   = Beta('asc_reca  ',0,-10,10,0,'taxes')
asc_tjcr   = Beta('asc_reca  ',0,-10,10,0,'cards')
asc_valo   = Beta('asc_valo  ',0,-10,10,0,'securities')
asc_nomina = Beta('asc_nomina',0,-10,10,0,'payroll')
asc_recibo = Beta('asc_recibo',0,-10,10,1,'direct db.')

w_other = Beta('w_other',0.5,0,1,0, 'class prob.')

asc = [asc_aval, asc_deco, asc_deme, asc_dela, asc_fond, asc_hip, asc_plan, asc_pres, asc_reca, asc_tjcr, asc_valo, asc_nomina, asc_recibo]

b_age   = [Beta('b_age'   + format(n, '02d'),0,-10,10,0,'age')
           for n,ch in enumerate(asc)]
b_loyal = [Beta('b_loyal' + format(n, '02d'),0,-10,10,0,'loyalty')
           for n,ch in enumerate(asc)]
b_sex   = [Beta('b_sex'   + format(n, '02d'),0,-10,10,0,'sex')
           for n,ch in enumerate(asc)]
b_empl  = [Beta('b_empl'  + format(n, '02d'),0,-10,10,0,'employee')
           for n,ch in enumerate(asc)]
b_activ = [Beta('b_activ' + format(n, '02d'),0,-10,10,0,'active')
           for n,ch in enumerate(asc)]
b_newcs = [Beta('b_newcs' + format(n, '02d'),0,-10,10,0,'new cust.')
           for n,ch in enumerate(asc)]
b_resid = [Beta('b_resid' + format(n, '02d'),0,-10,10,0,'resident')
           for n,ch in enumerate(asc)]
b_forgn = [Beta('b_forgn' + format(n, '02d'),0,-10,10,0,'foreigner')
           for n,ch in enumerate(asc)]
b_eu    = [Beta('b_eu'    + format(n, '02d'),0,-10,10,0,'european')
           for n,ch in enumerate(asc)]
b_vip   = [Beta('b_vip'   + format(n, '02d'),0,-10,10,0,'vip')
           for n,ch in enumerate(asc)]
b_sav   = [Beta('b_sav'   + format(n, '02d'),0,-10,10,0,'savings')
           for n,ch in enumerate(asc)]
b_curr  = [Beta('b_curr'  + format(n, '02d'),0,-10,10,0,'current')
           for n,ch in enumerate(asc)]
b_deriv = [Beta('b_deriv' + format(n, '02d'),0,-10,10,0,'derivada')
           for n,ch in enumerate(asc)]
b_payr  = [Beta('b_payr'  + format(n, '02d'),0,-10,10,0,'payroll_acc')
           for n,ch in enumerate(asc)]
b_junr  = [Beta('b_junr'  + format(n, '02d'),0,-10,10,0,'junior_acc')
           for n,ch in enumerate(asc)]
b_masp  = [Beta('b_masp'  + format(n, '02d'),0,-10,10,0,'masparti')
           for n,ch in enumerate(asc)]
b_parti = [Beta('b_parti' + format(n, '02d'),0,-10,10,0,'particular')
           for n,ch in enumerate(asc)]
b_pplus = [Beta('b_pplus' + format(n, '02d'),0,-10,10,0,'partiplus')
           for n,ch in enumerate(asc)]
b_eacc  = [Beta('b_eacc'  + format(n, '02d'),0,-10,10,1,'e-account')
           for n,ch in enumerate(asc)]


# Utility functions

V11 = np.dot(european, b_eu) + np.dot(payroll_acc, b_payr) + asc

V21 = np.dot(active, b_activ) + np.dot(e_acc, b_eacc) + asc

# Associate utility functions with the numbering of alternatives
V1 = {n+1: p for n,p in enumerate(V11)}
V2 = {n+1: p for n,p in enumerate(V21)}
av = {n+1: 1 for n,p in enumerate(asc)}

# Class membership model
probClass1 = 1 - w_other
probClass2 = w_other

# The choice model is a discrete mixture of logit, with availability conditions
prob1 = bioLogit(V1, av, choice)
prob2 = bioLogit(V2, av, choice)
prob = probClass1 * prob1 + probClass2 * prob2

# Defines an itertor on the data
rowIterator('obsIter')

# DEfine the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(log(prob),'obsIter')

# All observations verifying the following expression will not be
# considered for estimation
# The modeler here has developed the model only for work trips.
# Observations such that the dependent variable CHOICE is 0 are also removed.
# exclude = (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) + ( CHOICE == 0 )) > 0
# BIOGEME_OBJECT.EXCLUDE = exclude

# Statistics

nullLoglikelihood(av,'obsIter')
choiceSet = [1,2,3,4,5,6,7,8,9,10,11,12,13]
cteLoglikelihood(choiceSet,choice,'obsIter')
availabilityStatistics(av,'obsIter')


BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = 'BIO'
BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = '8'

# BIOGEME_OBJECT.FORMULAS['Train utility'] = V1
# BIOGEME_OBJECT.FORMULAS['Swissmetro utility'] = V2
# BIOGEME_OBJECT.FORMULAS['Car utility'] = V3
