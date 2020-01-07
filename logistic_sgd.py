from __future__ import print_function
import pickle, os, sys, timeit
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from optimizers import *
from utils import *
from neural_networks import *

def sgd_optimization(init_lr=0.1, dataset='santander.csv.h5', batch_size=64,
                     n_epochs=1000, in_size=20, out_size=13, decay=0.999,
                     momentum=0.5):
    """ Demonstrate stochastic gradient descent optimization
    :type init_lr: float
        :param init_lr: initial learning rate used
    :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
        :param dataset: the path of the dataset file
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0][0], datasets[0][1]-1
    valid_set_x, valid_set_y = datasets[1][0], datasets[1][1]-1
    test_set_x, test_set_y   = datasets[2][0], datasets[2][1]-1

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')
    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x')   # data
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    learning_rate = theano.shared(
        value=np.asarray(init_lr/batch_size, dtype=theano.config.floatX),
        borrow=True
    )

    # construct the logistic regression class
    mnl = LogisticRegression(input=x, output=y, n_in=in_size, n_out=out_size)
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    negLogLikelihood = mnl.negative_log_likelihood(y)
    predError = mnl.errors(y)

    # compute the gradient of cost with respect to theta = (W,b)
    grads = T.grad(negLogLikelihood, mnl.params)
    opt = sgd_nesterov(mnl.params)
    updates = opt.updates(mnl.params, grads, learning_rate, momentum)

    # construct Theano functions
    update_learning_rate = theano.function(
        inputs=[], outputs=learning_rate,
        updates={
            learning_rate: T.clip(
                learning_rate * decay, init_lr/batch_size * 0.01, 1)
        }
    )

    # compile a predictor function
    predict_model = theano.function([x], mnl.y_pred)
    loglikelihood = theano.function([x, y], negLogLikelihood)

    train_model = theano.function(
        inputs=[index], outputs=negLogLikelihood, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    valid_model = theano.function(
        inputs=[index], outputs=predError,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index], outputs=predError,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 400 * n_train_batches # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0

    #calculate null loglikelihood
    mnl.nullLogLikelihood = loglikelihood(
        train_set_x.get_value(borrow=True), train_set_y.eval()
    ) * train_set_x.get_value(borrow=True).shape[0]
    print(mnl.nullLogLikelihood)
    # we need to divide this by the batch size, since the actual L is
    # averaged across the minibatch

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        minibatch_avg_cost = 0
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost += train_model(minibatch_index) * batch_size
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                valid_losses = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(valid_losses)
                this_negLogLikelihood = np.mean(minibatch_avg_cost)

                print(
                    'epoch %i, minibatch %i/%i, negLogLikelihood %f, '
                    'validation score %f %%' %
                    (
                        epoch, minibatch_index + 1, n_train_batches,
                        this_negLogLikelihood, this_validation_loss * 100.
                    )
                )
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                    improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_losses= [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print('test score %f %%' % (test_score * 100.))

                    # prediction
                    predict_val = predict_model(test_set_x.get_value(borrow=True)) + 1
                    print(predict_val)
                    print(np.unique(predict_val, return_counts=True))

                    mnl.finalLogLikelihood = loglikelihood(
                        train_set_x.get_value(borrow=True), train_set_y.eval()
                    ) * train_set_x.get_value(borrow=True).shape[0]

                    # save the best model
                    with open('mnl_model.pkl', 'wb') as f:
                        pickle.dump(mnl, f)
                else:
                    update_learning_rate()

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete with best validation error of %f %%,'
          'with final log likelihood %f, null loglikelihood %f' %
        (
            best_validation_loss * 100.,
            -mnl.finalLogLikelihood,
            -mnl.nullLogLikelihood
        )
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def predict():
    """ An example of how to load a trained model and use it
    to predict labels.
    """
    # load the saved model
    mnl = pickle.load(open('mnl_model.pkl', 'rb'))

    # load the data
    datasets = load_data('santander.csv.h5')
    test_set_x, test_set_y = datasets[2][0], datasets[2][1]-1

    # compile a predictor function
    predict_model = theano.function([mnl.input], mnl.y_pred)

    # predict
    prediction = predict_model(test_set_x.eval())

def analytics():
    """ Generates analytical data from model
    """
    np.set_printoptions(precision=4, suppress=True)

    # load the saved model
    mnl = pickle.load(open('mnl_model.pkl', 'rb'))

    # load the data
    datasets = load_data('santander.csv.h5')
    train_set_x, train_set_y = datasets[0][0], datasets[0][1]-1
    valid_set_x, valid_set_y = datasets[1][0], datasets[1][1]-1
    test_set_x, test_set_y   = datasets[2][0], datasets[2][1]-1

    # compile the hessian function
    print('... compiling Hessians')
    hessian = theano.gradient.hessian(
        cost=mnl.negative_log_likelihood(mnl.output),
        wrt=mnl.params
    )
    hessian_fn = theano.function([mnl.input, mnl.output], hessian)

    # solve for Hessian
    print('... solving Hessians')
    solve = hessian_fn(test_set_x.eval(), test_set_y.eval())

    # evaluate t-stats
    SE_W, SE_b = [np.sqrt(np.diag(mat)) for mat in solve]
    t_stat_W = mnl.W.eval() / SE_W
    t_stat_b = mnl.b.eval() / SE_b
    t_stat = np.concatenate((t_stat_W, t_stat_b)).reshape(valid_set_x.get_value(borrow=True).shape[1]+1,-1)
    print(t_stat)

    # Hinton diagrams
    ylabels = [
        'guarantees', 'short term deposits', 'medium term deposits',
        'long term deposits', 'funds', 'mortgage', 'pensions', 'loans',
        'taxes', 'cards', 'securities', 'payroll','direct debit'
    ]
    xlabels = [
        'age', 'loyalty', 'income', 'sex', 'employee', 'active', 'new_cust',
        'resident', 'foreigner', 'european', 'vip', 'savings', 'current',
        'derivada', 'payroll_acc', 'junior', 'masparti', 'particular',
        'partiplus', 'e_acc', 'constant'
    ]
    hinton_matrix = np.concatenate((mnl.W_mat.eval(), [mnl.b.eval()]))
    ax = hinton(hinton_matrix, t_stat)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation="vertical")
    plt.title('title')
    plt.show()

if __name__ == '__main__':
    sgd_optimization()
