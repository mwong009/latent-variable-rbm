# Santander python script

from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *

#Parameters to be estimated
# Arguments:
#   - 1  Name for report; Typically, the same as the variable.
#   - 2  Starting value.
#   - 3  Lower bound.
#   - 4  Upper bound.
#   - 5  0: estimate the parameter, 1: keep it fixed.
#
ASC_ECUE = Beta('ASC_ECUE',0,-10,10,0,'e-account cte.')
ASC_FOND = Beta('ASC_FOND',0,-10,10,0,'Funds cte.')
ASC_HIP = Beta('ASC_HIP',0,-10,10,0,'Mortgage cte.')
ASC_PLAN = Beta('ASC_PLAN',0,-10,10,0,'Pensions cte.')
ASC_PRES = Beta('ASC_PRES',0,-10,10,0,'Loans cte.')
ASC_RECA = Beta('ASC_RECA',0,-10,10,0,'Taxes cte.')
ASC_TJCR = Beta('ASC_TJCR',0,-10,10,1,'Credit Card cte.')
ASC_VALO = Beta('ASC_VALO',0,-10,10,0,'Securities cte.')
ASC_VIV = Beta('ASC_VIV',0,-10,10,0,'Home Account cte.')
ASC_NOMINA = Beta('ASC_NOMINA',0,-10,10,0,'Payroll cte.')
ASC_NOMPENS = Beta('ASC_NOMPENS',0,-10,10,0,'Pensions cte.')
ASC_RECIBO = Beta('ASC_RECIBO',0,-10,10,0,'Direct Debit cte.')

B_SEX = Beta('B_SEX',0,-10,10,0,'Sex')
B_AGE = Beta('B_AGE',0,-10,10,0,'Age')
B_MON = Beta('B_MON',0,-10,10,0,'Month') # CATEGORIAL

B_LOYAL = Beta('B_LOYAL',0,-10,10,0,'Loyalty')
B_INC = Beta('B_INC',0,-10,10,0,'Income')
B_RESI = Beta('B_RESI',0,-10,10,0,'Resident')
B_FOREIGN = Beta('B_FOREIGN',0,-10,10,0,'Foreigner')
B_PROV = Beta('B_PROV',0,-10,10,0,'') # CATEGORIAL
B_COUNTRY = Beta('B_COUNTRY',0,-10,10,0,'') # CATEGORIAL
B_REGION = Beta('B_REGION',0,-10,10,0,'') # CATEGORIAL
B_SUBREGION = Beta('B_SUBREGION',0,-10,10,0,'') # CATEGORIAL
B_VIP = Beta('B_VIP',0,-10,10,0,'VIP')
B_INDV = Beta('B_INDV',0,-10,10,0,'Individual')
B_GRAD = Beta('B_GRAD',0,-10,10,0,'Graduate')

B_SAV = Beta('B_SAV',0,-10,10,0,'Savings')
B_GUARAN = Beta('B_GUARAN',0,-10,10,0,'Guarantees')
B_CURR = Beta('B_CURR',0,-10,10,0,'Current')
B_DERIV = Beta('B_DERIV',0,-10,10,0,'Derivatives')
B_PAYR = Beta('B_PAYR',0,-10,10,0,'Payroll')
B_JUNIOR = Beta('B_JUNIOR',0,-10,10,0,'Junior')
B_MASPART = Beta('B_MASPART',0,-10,10,0,'Mas Particular')
B_PARTI = Beta('B_PARTI',0,-10,10,0,'Particular')
B_PARTIPLUS = Beta('B_PARTIPLUS',0,-10,10,0,'Particular Plus')
B_SDEP = Beta('B_SDEP',0,-10,10,0,'Short term deposits')
B_MDEP = Beta('B_MDEP',0,-10,10,0,'Medium term depositis')
B_LDEP = Beta('B_LDEP',0,-10,10,0,'Long term deposits')

# Utility functions

# For numerical reasons, it is good practice to scale the data to
# that the values of the parameters are around 1.0.
# A previous estimation with the unscaled data has generated
# parameters around -0.01 for both cost and time. Therefore, time and
# cost are multipled my 0.01.

# The following statements are designed to preprocess the data. It is
# like creating a new columns in the data file. This should be
# preferred to the statement like
# TRAIN_TT_SCALED = TRAIN_TT / 100.0
# which will cause the division to be reevaluated again and again,
# throuh the iterations. For models taking a long time to estimate, it
# may make a significant difference.

INCOME_SCALED = DefineVariable(
	'INCOME_SCALED', income / 1000000.0 if income != 99999 else 99999)
B_AGE1 = Beta('B_AGE1',0,-10,10,0,'Age 1')
AGE_1 = DefineVariable('AGE_1', age < 18)
B_AGE2 = Beta('B_AGE2',0,-10,10,0,'Age 2')
AGE_2 = DefineVariable('AGE_2', (age >= 18) + (age < 22))
B_AGE3 = Beta('B_AGE3',0,-10,10,0,'Age 3')
AGE_3 = DefineVariable('AGE_3', (age >= 22) + (age < 26))
B_AGE4 = Beta('B_AGE4',0,-10,10,0,'Age 4')
AGE_4 = DefineVariable('AGE_4', (age >= 26) + (age < 30))
B_AGE5 = Beta('B_AGE5',0,-10,10,0,'Age 5')
AGE_5 = DefineVariable('AGE_5', (age >= 30) + (age < 34))
B_AGE6 = Beta('B_AGE6',0,-10,10,0,'Age 6')
AGE_6 = DefineVariable('AGE_6', (age >= 34) + (age < 38))
B_AGE7 = Beta('B_AGE7',0,-10,10,0,'Age 7')
AGE_7 = DefineVariable('AGE_7', (age >= 38) + (age < 42))
B_AGE8 = Beta('B_AGE8',0,-10,10,0,'Age 8')
AGE_8 = DefineVariable('AGE_8', age >= 42)

B_JAN = Beta('B_JAN',0,-10,10,0,'January')
B_FEB = Beta('B_FEB',0,-10,10,0,'February')
B_MAR = Beta('B_MAR',0,-10,10,0,'March')
B_APR = Beta('B_APR',0,-10,10,0,'April')
B_MAY = Beta('B_MAY',0,-10,10,0,'May')
B_JUN = Beta('B_JUN',0,-10,10,0,'June')
B_JUL = Beta('B_JUL',0,-10,10,0,'July')
B_AUG = Beta('B_AUG',0,-10,10,0,'August')
B_SEP = Beta('B_SEP',0,-10,10,0,'September')
B_OCT = Beta('B_OCT',0,-10,10,1,'October')
B_NOV = Beta('B_NOV',0,-10,10,0,'November')
B_DEC = Beta('B_DEC',0,-10,10,0,'December')
MON_JAN = DefineVariable('MON_JAN', mon == 1)
MON_FEB = DefineVariable('MON_FEB', mon == 2)
MON_MAR = DefineVariable('MON_MAR', mon == 3)
MON_APR = DefineVariable('MON_APR', mon == 4)
MON_MAY = DefineVariable('MON_MAY', mon == 5)
MON_JUN = DefineVariable('MON_JUN', mon == 6)
MON_JUL = DefineVariable('MON_JUL', mon == 7)
MON_AUG = DefineVariable('MON_AUG', mon == 8)
MON_SEP = DefineVariable('MON_SEP', mon == 9)
MON_OCT = DefineVariable('MON_OCT', mon == 10)
MON_NOV = DefineVariable('MON_NOV', mon == 11)
MON_DEC = DefineVariable('MON_DEC', mon == 12)

V1 = ASC_ECUE
V2 = ASC_FOND
V3 = ASC_HIP
V4 = ASC_PLAN
V5 = ASC_PRES
V6 = ASC_RECA
V7 = ASC_TJCR + B_JAN * MON_JAN + B_FEB * MON_FEB + B_MAR * MON_MAR \
			+ B_APR * MON_APR + B_MAY * MON_MAY + B_JUN * MON_JUN \
			+ B_JUL * MON_JUL + B_AUG * MON_AUG + B_SEP * MON_SEP \
			+ B_OCT + MON_OCT + B_NOV * MON_NOV + B_DEC * MON_DEC \
			#+ B_MON * mon
			#+ B_AGE2 * AGE_2 + B_AGE3 * AGE_3 \
			#+ B_AGE4 * AGE_4 + B_AGE5 * AGE_5 + B_AGE6 * AGE_6 \
			#+ B_AGE7 + AGE_7 + B_AGE8 * AGE_8
			#+ B_AGE * age
			#+ B_SEX * sex + B_LOYAL * loyalty \
			  #+ B_RESI * residence + B_FOREIGN * foreigner \
			  #+ B_VIP * vip + B_GRAD * grad \
			  #+ B_SAV * savings + B_CURR * current \
			  # + B_MON * mon + B_PROV * province \
			  # + B_AGECAT * AGE_CATEGORIAL \
			  #+ B_INC * INCOME_SCALED \
			  #+ B_DERIV * derivada + B_PAYR * payroll
			  #+ B_PARTI * particular \
			  #+ B_LDEP * longdep
V8 = ASC_VALO
V9 = ASC_VIV
V10 = ASC_NOMINA
V11 = ASC_NOMPENS
V12 = ASC_RECIBO

# Associate utility functions with the numbering of alternatives
V = {1: V1,
	 2: V2,
	 3: V3,
	 4: V4,
	 5: V5,
	 6: V6,
	 7: V7,
	 8: V8,
	 9: V9,
	 10: V10,
	 11: V11,
	 12: V12
	}

# Associate the availability conditions with the alternatives
# CAR_AV_SP =  DefineVariable('CAR_AV_SP',CAR_AV  * (  SP   !=  0  ))
# TRAIN_AV_SP =  DefineVariable('TRAIN_AV_SP',TRAIN_AV  * (  SP   !=  0  ))

av = {1: 1,
	  2: 1,
	  3: 1,
	  4: 1,
	  5: 1,
	  6: 1,
	  7: 1,
	  8: 1,
	  9: 1,
	  10: 1,
	  11: 1,
	  12: 1}

# The choice model is a logit, with availability conditions
logprob = bioLogLogit(V,av,choice)

# Defines an itertor on the data
rowIterator('obsIter')

# DEfine the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')

# All observations verifying the following expression will not be
# considered for estimation
# The modeler here has developed the model only for work trips.
# Observations such that the dependent variable CHOICE is 0 are also removed.
exclude = (( choice == 1 )) > 0

#BIOGEME_OBJECT.EXCLUDE = exclude

# Statistics

#nullLoglikelihood(av,'obsIter')
#choiceSet = [1,7]
#cteLoglikelihood(choiceSet,choice,'obsIter')
#availabilityStatistics(av,'obsIter')


BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "CFSQP"
BIOGEME_OBJECT.PARAMETERS['numberOfThreads'] = "16"

BIOGEME_OBJECT.FORMULAS['Credit Card utility'] = V1
