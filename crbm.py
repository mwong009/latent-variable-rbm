import timeit, pickle, sys, os
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import *
from neural_networks import *
from optimizers import *


class CRBM(object):
    """Conditional Restricted Boltzmann Machine (CRBM)"""
    def __init__(
        self, input=None, input_context=None,
        n_visible=None, n_hidden=None, n_context=None,
        W=None, U=None, V=None, latent_sample=None,
        hbias=None, vbias=None, np_rng=None,
        theano_rng=None):
        """
        CRBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.
        :param input:       None for standalone RBMs or symbolic variable if
                            RBM is part of a larger graph.

        :param n_visible:   number of visible units (target)
        :param n_hidden:    number of hidden units
        :param n_context:   number of context variables

        :param U:   None for standalone CRBMs or symbolic variable pointing to a
                    shared weight matrix in case CRBM is part of a CDBN
                    network; in a CDBN, the weights are shared between CRBMs
                    and layers of a MLP

        :param V:   None for standalone CRBMs or symbolic variable pointing to a
                    shared weight matrix in case CRBM is part of a CDBN
                    network; in a CDBN, the weights are shared between CRBMs
                    and layers of a MLP

        :param W:   None for standalone CRBMs or symbolic variable pointing to a
                    shared weight matrix in case CRBM is part of a CDBN
                    network; in a CDBN, the weights are shared between CRBMs
                    and layers of a MLP

        :param hbias:   None for standalone CRBMs or symbolic variable pointing
                        to a shared hidden units bias vector in case CRBM is
                        part of a different network

        :param vbias:   None for standalone RBMs or a symbolic variable
                        pointing to a shared visible units bias
        """
        self.input_context = input_context
        self.input = input
        self.latent_sample = latent_sample

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_context = n_context
        self.nullLogLikelihood = None
        self.finalLoglikelihood = None


        if np_rng is None:
            # create a number generator
            np_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        # initialize input layer for standalone CRBM or layer0 of CDBN

        if W is None:
            W = theano.shared(
                value=np.asarray(
                    np_rng.uniform(
                        low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                        high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                        size=(n_visible*n_hidden)
                    ),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )

        if U is None:
            U = theano.shared(
                value=np.asarray(
                    np_rng.uniform(
                        low=-4 * np.sqrt(6. / (n_context + n_visible)),
                        high=4 * np.sqrt(6. / (n_context + n_visible)),
                        size=(n_context*n_visible)
                    ),
                    dtype=theano.config.floatX
                ),
                name='U',
                borrow=True
            )

        if V is None:
            V = theano.shared(
                value=np.asarray(
                    np_rng.uniform(
                        low=-4 * np.sqrt(6. / (n_hidden + n_context)),
                        high=4 * np.sqrt(6. / (n_hidden + n_context)),
                        size=(n_context*n_hidden)
                    ),
                    dtype=theano.config.floatX
                ),
                name='V',
                borrow=True
            )

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                name='vbias',
                borrow=True
            )

        self.W = W
        self.W_mat = self.W.reshape((n_visible, n_hidden))
        self.U = U
        self.U_mat = self.U.reshape((n_context, n_visible))
        self.V = V
        self.V_mat = self.V.reshape((n_context, n_hidden))
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias, self.U, self.V]

        p_h_given_x_mean = T.nnet.sigmoid(
            T.dot(self.input_context, self.V_mat) + self.hbias
        )
        self.p_h_given_x = self.theano_rng.binomial(
            size=p_h_given_x_mean.shape,
            n=1,
            p=p_h_given_x_mean,
            dtype=theano.config.floatX
        )

        self.p_y_given_xh = T.nnet.softmax(
            T.dot(p_h_given_x_mean, self.W_mat.T) +
            T.dot(self.input_context, self.U_mat) + self.vbias
        )

        self.y_pred = T.argmax(self.p_y_given_xh, axis=1)

    def free_energy(self, v_sample, v_context):
        '''' Function to compute the free energy of a sample conditional
        on the context '''
        wx_b = T.dot(v_sample, self.W_mat) + \
               T.dot(v_context, self.V_mat) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias) + T.dot(T.dot(v_context, self.U_mat), v_sample.T)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return - hidden_term - vbias_term

        # wx_b = T.dot(v_sample, self.W_mat) + T.dot(v_context, self.V_mat) + self.hbias
        # ax_b = T.dot(v_context, self.U_mat) + self.vbias
        # visible_term = T.sum(0.5 * T.sqr(v_sample - ax_b), axis=1)
        # hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        # return visible_term - hidden_term


    def propup(self, vis, con):
        '''This function propagates the visible units activation upwards to
        the hidden units
        Note that we return also the pre-sigmoid activation
        '''
        pre_sigmoid_activation = T.dot(vis, self.W_mat) + \
                                 T.dot(con, self.V_mat) + \
                                 self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample, v0_context):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample, v0_context)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = h1_mean
        # h1_sample = self.theano_rng.binomial(
        #     size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)

        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid, con):
        '''This function propagates the hidden units activation downwards to
        the visible units
        Note that we return also the pre_softmax_activation
        '''
        pre_sigmoid_activation = T.dot(hid, self.W_mat.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample, v0_context):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample, v0_context)
        # get a sample of the visible given their activation
        # Note that theano_rng.multinomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(
            size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, v0_context):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(
            h0_sample, v0_context)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(
            v1_sample, v0_context)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample, v0_context):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(
            v0_sample, v0_context)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(
            h1_sample, v0_context)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k

        :param lr: learning rate used to train the RBM

        :param persistent: None for CD. For PCD, shared variable
            containing old state of Gibbs chain. This must be a shared
            variable of size (batch size, number of hidden units).

        :param k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The
        dictionary contains the update rules for weights and biases but
        also an update of the shared variable used to store the persistent
        chain, if one is used.

        """
        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(
            T.extra_ops.to_one_hot(self.input, 13), self.input_context)

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        (
            [pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates
        ) = theano.scan(
            self.gibbs_hvh,
            # the None are place holders, saying that
            # chain_start is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, chain_start],
            non_sequences=self.input_context,
            n_steps=k
        )

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]

        gen_cost = T.mean(
                    self.free_energy(
                        T.extra_ops.to_one_hot(self.input,13),
                        self.input_context
                    )
                ) - \
                T.mean(
                    self.free_energy(
                        chain_end,
                        self.input_context
                    )
                )

        disc_cost = self.negative_log_likelihood(self.input)

        # We must not compute the gradient through the gibbs sampling
        grads = T.grad(
            0.01*gen_cost+disc_cost, self.params,
            consider_constant=[chain_end]
        )

        # constructs the update dictionary
        # for grad, param in zip(grads, self.params):
        #     # make sure that the learning rate is of the right dtype
        #     updates[param] = param - grad * T.cast(
        #         lr, dtype=theano.config.floatX)
        opt = sgd_nesterov(self.params)
        updates = opt.updates(self.params, grads, lr, 0.5)

        monitoring_cost = self.get_reconstruction_cost(
            updates, pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""

        # index of bit i in expression p(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi, self.input_context)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:,bit_i_idx] = 1-xi[:, bit_i_idx], but assigns
        # the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip, self.input_context)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip -
                                                            fe_xi)))

        # increment bit_i_idx % number as part of updates
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible

        return cost

    def get_reconstruction_cost(self, updates, pre_softmax_nv):
        """Approximation to the reconstruction error """

        cross_entropy = T.nnet.binary_crossentropy(
            T.clip(T.nnet.softmax(pre_softmax_nv), 1e-6, 1.0-(1e-6)),
            T.extra_ops.to_one_hot(self.input,13)).mean()
        return cross_entropy

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood"""
        return -T.mean(T.log(self.p_y_given_xh)[T.arange(y.shape[0]), y])

    def cross_entropy(self, y):
        cross_entropy = T.nnet.binary_crossentropy(
            T.clip(self.p_y_given_xh, 1e-6, 1.0-(1e-6)),
            T.extra_ops.to_one_hot(y,13)).mean()
        return cross_entropy

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch"""
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def pretraining_functions(self, train_set_x, train_set_y, batch_size, k):
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        pt_learning_rate = theano.shared(
            value=np.asarray(0.1, dtype=theano.config.floatX),
            borrow=True
        )

        update_ptlr = theano.function(
            inputs=[], outputs=pt_learning_rate,
            updates={ pt_learning_rate: T.clip(
                pt_learning_rate * 0.999, 0.1 / batch_size * 0.01, 1)
            }
        )

        cost, updates = self.get_cost_updates(learning_rate,
                                             persistent=None, k=k)

        fn = theano.function(
            inputs=[index, theano.In(learning_rate, value=pt_learning_rate.get_value())],
            outputs=cost,
            updates=updates,
            givens={
                self.input: train_set_y[
                    (index * batch_size):(index * batch_size + batch_size)],
                self.input_context: train_set_x[
                    (index * batch_size):(index * batch_size + batch_size)]
            }
        )

        return fn, update_ptlr

    def build_finetune_functions(self, datasets, batch_size, init_lr):

        train_set_x, train_set_y = datasets[0][0], datasets[0][1]-1
        valid_set_x, valid_set_y = datasets[1][0], datasets[1][1]-1
        test_set_x, test_set_y = datasets[2][0], datasets[2][1]-1

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = theano.shared(
            value=np.asarray(init_lr, dtype=theano.config.floatX),
            borrow=True
        )

        # compute the gradients with respect to the model parameters
        finetune_cost = self.negative_log_likelihood(self.input)
        grads = T.grad(finetune_cost, self.params)
        opt = sgd_nesterov(self.params)
        updates = opt.updates(self.params, grads, learning_rate, 0.5)

        # compute list of fine-tuning updates
        # updates = []
        # for param, gparam in zip(self.params, gparams):
        #     updates.append((param, param - gparam * learning_rate))

        loglikelihood_fn = theano.function(
            inputs=[index], outputs=finetune_cost,
            givens={
                self.input_context: train_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.input: train_set_y[
                    index * batch_size: (index + 1) * batch_size]
            }
        )

        train_fn = theano.function(
            inputs=[index], outputs=finetune_cost,
            updates=updates,
            givens={
                self.input_context: train_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.input: train_set_y[
                    index * batch_size: (index + 1) * batch_size]
            }
        )

        test_score_i = theano.function(
            inputs=[index], outputs=self.errors(self.input),
            givens={
                self.input_context: test_set_x[
                    index * batch_size: (index + 1) * batch_size],
                self.input: test_set_y[
                    index * batch_size: (index + 1) * batch_size]
            }
        )

        valid_score_i = theano.function(
            inputs=[index], outputs=self.errors(self.input),
            givens={
                self.input_context: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.input: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        update_lr = theano.function(
            inputs=[], outputs=learning_rate,
            updates={ learning_rate: T.clip(
                learning_rate * 0.999, init_lr / batch_size * 0.01, 1)
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score, update_lr, loglikelihood_fn

def test_crbm(init_lr=0.1, dataset='santander.csv.h5',
              batch_size=64, n_chains=10, n_samples=10, n_hidden=0,
              n_inputs=13, n_context=20, decay=0.999,
              training_epochs=200, pretraining_epochs=400):
    """
    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path the the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0][0], datasets[0][1]-1
    valid_set_x, valid_set_y = datasets[1][0], datasets[1][1]-1
    test_set_x, test_set_y = datasets[2][0], datasets[2][1]-1

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x= T.matrix('x')   # the data
    y= T.ivector('y')  # the labels are presented as matrix
    h= T.matrix('h') # latent samples

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # construct the RBM class
    crbm = CRBM(input=y, n_visible=n_inputs, n_hidden=n_hidden, latent_sample=h,
                input_context=x, n_context=n_context, np_rng=rng, theano_rng=theano_rng)

    # construct a predictor function
    pred_fn = theano.function(
        inputs=[crbm.input_context],
        outputs=crbm.y_pred
    )

    print('... getting the pretraining functions')
    pretraining_fns, update_ptlr = crbm.pretraining_functions(
        train_set_x=train_set_x,
        train_set_y=train_set_y,
        batch_size=batch_size, k=1)
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model, update_lr, ll_fn = crbm.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        init_lr=init_lr*0.01
    )

    #########################
    # PRETRAINING THE MODEL #
    #########################
    best_validation_loss = np.inf
    test_score = 0.
    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for epoch in range(pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns(index=batch_index, lr=0.1))
        print(
            'Pre-training RBM epoch %d, cost %f' %
            (epoch, np.mean(c, dtype='float64'))
        )
        validation_losses = validate_model()
        this_validation_loss = np.mean(
            validation_losses, dtype='float64'
        )
        print('epoch %i, validation error %f %%' % (
            epoch,
            this_validation_loss * 100.
            )
        )
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:

            # save best validation score and iteration number
            best_validation_loss = this_validation_loss

            modelfile = 'crbm%s_sub.pkl' % n_hidden
            best_model = crbm
            with open(modelfile, 'wb') as f:
                pickle.dump(best_model, f)

            predict = pred_fn(test_set_x.get_value(borrow=True))+1
            print(predict, np.unique(predict,return_counts=True))

            crbm.finalLoglikelihood = np.sum([ll_fn(i) for i in range(n_train_batches)]) * batch_size

            # test it on the test set
            test_losses = test_model()
            test_score = np.mean(test_losses, dtype='float64')
            print(('     epoch %i, test error of '
                   'best model %f %%') %
                  (epoch, test_score * 100.))
        else:
            update_ptlr()


    end_time = timeit.default_timer()

    print('The pretraining code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

    ########################
    # FINETUNING THE MODEL #
    ########################
    print('... finetuning the model')
    crbm = best_model
    # early-stopping parameters
    # look as this many examples regardless
    patience = 400 * n_train_batches
    # wait this much longer when a new best is found
    patience_increase = 2.
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.999

    validation_frequency = min(n_train_batches, patience / 2)
    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(
                    validation_losses, dtype='float64'
                )
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                    )
                )
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                            improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                          test_score * 100.))

                    modelfile = 'crbm%s_sub.pkl' % n_hidden
                    with open(modelfile, 'wb') as f:
                        pickle.dump(crbm, f)

                    predict = pred_fn(test_set_x.get_value(borrow=True))+1
                    print(predict, np.unique(predict,return_counts=True))

                    crbm.finalLoglikelihood = np.sum([train_fn(i) for i in range(n_train_batches)]) * batch_size

                else:
                    update_lr()
                    # go through the training set
                    # c = []
                    # for batch_index in range(n_train_batches):
                    #     c.append(pretraining_fns(index=batch_index, lr=0.1))
                    # print(
                    #     'optimizing RBM epoch %d, K-L loss %f' %
                    #     (epoch, np.mean(c, dtype='float64'))
                    # )

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           'with test performance %f %%'
           ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

    print (
        'Optimization complete with best validation error of %f %%,'
        'with final log likelihood %f' %
        (
            best_validation_loss * 100.,
            -crbm.finalLoglikelihood
        )
    )
    print('hyperparameters: hidden units %s' % n_hidden)

def predict():
    """ An example of how to load a trained model and use it
    to predict labels.
    """
    # load the saved model
    crbm = pickle.load(open('crbm16.pkl', 'rb'))

    # load the data
    datasets = load_data('santander.csv.h5')
    test_set_x, test_set_y = datasets[2][0], datasets[2][1]-1

    # compile a predictor function
    predict_model = theano.function([crbm.input_context], crbm.y_pred)

    # predict
    prediction = predict_model(test_set_x.get_value(borrow=True))
    print(prediction)
    print(np.unique(prediction, return_counts=True))

if __name__ == '__main__':
    test_crbm()
