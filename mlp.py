from __future__ import print_function

__docformat__ = 'restructedtext en'

import pickle
import h5py
import gzip
import os, sys, timeit
import numpy
import theano
import theano.tensor as T
from neural_networks import *
from utils import *
from optimizers import *

# start-snippet-2


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.000, n_epochs=1000,
             dataset='santander.csv.h5', batch_size=100, n_hidden=4,
             in_size=20, out_size=13):
    """ multilayer perceptron
    :type learning_rate: float
    :param learning_rate: learning rate used

    :type L1_reg: float
    :param L1_reg: L1-norm's weight
    :type L2_reg: float
    :param L2_reg: L2-norm's weight
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer
    :type dataset: string
    :param dataset: the path of the dataset file
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0][0], datasets[0][1]-1
    valid_set_x, valid_set_y = datasets[1][0], datasets[1][1]-1
    test_set_x, test_set_y = datasets[2][0], datasets[2][1]-1

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=in_size,
        n_hidden=n_hidden,
        n_out=out_size
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    nll = (classifier.negative_log_likelihood(y))

    # compute the gradient of cost with respect to theta = (W,b)
    grads = T.grad(nll, classifier.params)
    # opt = sgd(classifier.params)
    # updates = opt.updates(classifier.params, grads, learning_rate)
    opt = sgd_nesterov(classifier.params)
    updates = opt.updates(classifier.params, grads, learning_rate, 0.9)

    train_model = theano.function(
        inputs=[index],
        outputs=nll,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    loglikelihood = theano.function(
        inputs=[index],
        outputs=nll,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    hessians = theano.function(
        inputs=[index],
        outputs=theano.gradient.hessian(cost=nll, wrt=classifier.params),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training the model')
    # early-stopping parameters
    patience = 1000 * n_train_batches  # look as this many examples regardless
    patience_increase = 16  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    #calculate Null log-likelihood
    null_ll = numpy.mean(
        [loglikelihood(i) for i in range(n_train_batches)]
    )
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                valid_losses = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(valid_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation score %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set
                    test_losses= [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                            '     epoch %i, minibatch %i/%i,'
                            ' test score %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # prediction
                    predict_val = predict_model(test_set_x.get_value())+1
                    print(predict_val)
                    print(numpy.unique(predict_val, return_counts=True))

                    # save the best model
                    best_model = classifier
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break
    # hessians
    classifier = best_model
    W_h, b_h = numpy.mean(
        [hessians(i) for i in range(n_train_batches)]
        , axis=0)
    SE = numpy.diag(
        numpy.sqrt(1/W_h))
    t_stat = (classifier.W.eval() / SE)
    numpy.set_printoptions(precision=5, suppress=True)
    print(SE)
    print(numpy.vstack((classifier.W.eval(),t_stat)))

    final_ll = numpy.mean(
        [loglikelihood(i) for i in range(n_train_batches)]
    )
    r_square = 1- (final_ll/null_ll)
    print('r_sqr: %f' % r_square)

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            ' with negative log likelihood -%f'
        )
        % (best_validation_loss * 100., test_score * batch_size)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


if __name__ == '__main__':
    test_mlp()
