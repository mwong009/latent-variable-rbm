import timeit, pickle, sys, os
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import *
from neural_networks import *
from optimizers import *

from crbm import CRBM

def analytics():
    """ Generates analytical data from model
    """
    np.set_printoptions(precision=4, suppress=True)
    # load the saved model
    print('... loading models')
    modelfile = [
        'mnl_model.pkl',
        'crbm2_sub.pkl',
        'crbm4_sub.pkl',
        'crbm8_sub.pkl',
        'crbm16_sub.pkl'
    ]
    model = pickle.load(open(modelfile[0], 'rb'))


    # load the data
    datasets = load_data('santander.csv.h5')
    train_set_x, train_set_y = datasets[0][0], datasets[0][1]-1
    valid_set_x, valid_set_y = datasets[1][0], datasets[1][1]-1
    test_set_x, test_set_y   = datasets[2][0], datasets[2][1]-1

    n_samples = train_set_x.get_value(borrow=True).shape[0]
    n_in = 20
    n_out = 13
    n_hidden = 0

    # compile the hessian function
    print('... compiling Hessians')
    loglikelihood_cost = model.negative_log_likelihood(model.input)
    hessian = theano.gradient.hessian(cost=loglikelihood_cost, wrt=model.params)
    hessian_fn = theano.function(
        inputs=[model.input_context, model.input],
        outputs=hessian)

    # solve for Hessian
    print('... solving Hessians')
    solve = hessian_fn(
        train_set_x.get_value(borrow=True), train_set_y.eval())

    # compiling statistics
    print('... compiling statistics')
    se = {}
    standard_errors = [np.sqrt(2/((n_samples-1)*np.diag(mat))) for mat in solve]
    for param, standard_error in zip(model.params, standard_errors):
        p = {}
        t_stat = np.array(param.eval() / standard_error)
        p['params'] = np.array(param.eval())
        p['se'] = standard_error
        p['tstat'] = t_stat
        se[param.name] = p

    # for crbm
    se['W']['params'] = se['W']['params'].reshape(n_out,n_hidden).T.flatten()
    se['W']['se'] = se['W']['se'].reshape(n_out,n_hidden).T.flatten()
    se['W']['tstat'] = se['W']['tstat'].reshape(n_out,n_hidden).T.flatten()

    # print standard errors
    print('standard errors:')
    np.savetxt('se_lv.csv', np.concatenate([se[k]['se'] for k in ('U', 'W', 'vbias', 'V', 'hbias')]), delimiter=',')

    # merge matrices
    # mat_tst = np.concatenate([se[k]['tstat'].reshape(-1, n_out) for k in ('U', 'W', 'vbias')])
    # mat_hin = np.concatenate([se[k]['params'].reshape(-1, n_out) for k in ('U', 'W', 'vbias')])

    # # Hinton diagrams
    # ylabels = np.array(
    #     ['guarantees', 'short term deposits', 'medium term deposits',
    #     'long term deposits', 'funds', 'mortgage', 'pensions', 'loans',
    #     'taxes', 'cards', 'securities', 'payroll','direct debit'],
    #     dtype='U'
    # )
    # xlabels = np.array(np.hstack((
    #     ['age', 'loyalty', 'income', 'sex', 'employee', 'active', 'new_cust',
    #     'resident', 'foreigner', 'european', 'vip', 'savings', 'current',
    #     'derivada', 'payroll_acc', 'junior', 'masparti', 'particular',
    #     'partiplus', 'e_acc'],
    #     ['hidden%s' % i for i in range(1,n_hidden+1)],
    #     ['constant'])), dtype='U')
    #
    # plt.figure(figsize=(8,5.5))
    # ax = hinton(mat_hin, mat_tst)
    # ax.set_yticks(range(len(ylabels)))
    # ax.set_yticklabels(ylabels)
    # ax.set_xticks(range(len(xlabels)))
    # ax.set_xticklabels(xlabels, rotation="vertical")
    #
    # plt.title('C-RBM-16 model')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.clf()
    ######################
    #
    # xlabels = np.array(
    #     ['age', 'loyalty', 'income', 'sex', 'employee', 'active', 'new_cust',
    #     'resident', 'foreigner', 'european', 'vip', 'savings', 'current',
    #     'derivada', 'payroll_acc', 'junior', 'masparti', 'particular',
    #     'partiplus', 'e_acc', 'constant'], dtype='U'
    # )
    # ylabels = np.array(
    #     ['hidden%s' % i for i in range(1,n_hidden+1)], dtype='U'
    # )
    # mat_hin = np.concatenate([se[k]['params'].reshape(-1, n_hidden) for k in ('V', 'hbias')])
    # mat_tst = np.concatenate([se[k]['tstat'].reshape(-1, n_hidden) for k in ('V', 'hbias')])
    #
    # plt.figure(figsize=(8,5.5))
    # ax = hinton(mat_hin, mat_tst)
    # ax.set_yticks(range(len(ylabels)))
    # ax.set_yticklabels(ylabels)
    # ax.set_xticks(range(len(xlabels)))
    # ax.set_xticklabels(xlabels, rotation="vertical")
    #
    # plt.title('C-RBM-2 latent variables')
    # plt.show()
    #
    # plt.clf()
