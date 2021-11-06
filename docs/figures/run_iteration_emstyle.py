from src.util import data_load
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import gpflow
from src.cvi_svgp import SVGP_CVI
from sklearn.preprocessing import StandardScaler
import time
from gpflow.optimizers import NaturalGradient
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from scipy.io import loadmat

# Define parameters
n_e_steps = 8
n_m_steps = 20
nat_lr = 0.8
adam_lr = 0.1
M = 50
nm = 3 # number of models [svgp, svgp_nat, t-svgp]
nit = 20
t_nit = n_e_steps*nit + n_m_steps*nit

mb_size = 'full'
n_folds = 5

data_name = 'airfoil' #'boston', 'concrete', 'airfoil'
optim = 'Adam'

rng = np.random.RandomState(19)
tf.random.set_seed(19)


def init_model(n_train):
    models = []
    names = []

    # Define standard SVGP
    m = gpflow.models.SVGP(
        kernel=gpflow.kernels.Matern52(lengthscales=np.ones((1,x.shape[1]))*ell,variance=var),
        likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=Z.copy(), num_data=n_train)


    models.append(m)
    names.append('svgp')

    # Define natgrad SVGP
    m_svgp_nat = gpflow.models.SVGP(
        kernel=gpflow.kernels.Matern52(lengthscales=np.ones((1,x.shape[1]))*ell,variance=var),
        likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=Z.copy()
        ,num_data=n_train,whiten=True)

    gpflow.set_trainable(m_svgp_nat.q_mu , False)
    gpflow.set_trainable(m_svgp_nat.q_sqrt , False)


    models.append(m_svgp_nat)
    names.append('svgp_nat')

    # Define non-whitened version
    m_svgp_nat_nw = gpflow.models.SVGP(
        kernel=gpflow.kernels.Matern52(lengthscales=np.ones((1,x.shape[1]))*ell,variance=var),
        likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=Z.copy()
        ,num_data=n_train,whiten=False)

    gpflow.set_trainable(m_svgp_nat_nw .q_mu , False)
    gpflow.set_trainable(m_svgp_nat_nw .q_sqrt , False)


    models.append(m_svgp_nat_nw)
    names.append('svgp_nat_nw')

    #Define CVI
    m_cvi = SVGP_CVI(
    kernel=gpflow.kernels.Matern52(lengthscales=np.ones((1,x.shape[1]))*ell,variance=var),
    likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=Z.copy()
    ,num_data=n_train)


    # Turn off natural params
    gpflow.set_trainable(m_cvi.lambda_1 , False)
    gpflow.set_trainable(m_cvi.lambda_2_sqrt , False)

    models.append(m_cvi)
    names.append('svgp_cvi')

    return models,names

def run_optim(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """

    # Create an Adam Optimizer action
    logf = []
    nlpd = []

    natgrad_opt = NaturalGradient(gamma=nat_lr)
    if optim == 'Adam':
        optimizer = tf.optimizers.Adam(adam_lr)

    elif optim == 'SGD':
        optimizer = tf.optimizers.SGD(adam_lr)

    optimizer2 = tf.optimizers.Adam(nat_lr)

    train_iter = iter(train_dataset.batch(mb_size))

    training_loss = model.training_loss_closure(train_iter, compile=True)

    #@tf.function
    def optimization_step_nat(training_loss,variational_params):
        natgrad_opt.minimize(training_loss, var_list=variational_params)

    @tf.function
    def optimization_step_cvi(model,training_loss):
        model.natgrad_step(*data, lr=nat_lr)

    @tf.function
    def optimization_step(model,training_loss,params):
        optimizer.minimize(training_loss, var_list=params)

    @tf.function
    def optimization_step2(model,training_loss,params):
        optimizer2.minimize(training_loss, var_list=params)

    for step in range(iterations):
        data = next(train_iter)

        if model.name=='svgp' and model.q_mu.trainable==False:
            variational_params = [(model.q_mu, model.q_sqrt)]

            for i in range(n_e_steps):

                optimization_step_nat(training_loss,variational_params)
                # elbo = model.maximum_log_likelihood_objective(data).numpy()
                # nl = -tf.reduce_mean(model.predict_log_density((xt,yt))).numpy()


            elbo = model.maximum_log_likelihood_objective(data).numpy()
            logf.append(elbo)

            nlpd.append(-tf.reduce_mean(model.predict_log_density((xt,yt))).numpy())


            # for var in optimizer.variables():
            #     var.assign(tf.zeros_like(var))

            for j in range(n_m_steps):
                optimization_step(model,training_loss,model.trainable_variables)


        elif model.name=='svgp_cvi':

            for i in range(n_e_steps):
                optimization_step_cvi(model,training_loss)

            elbo = model.maximum_log_likelihood_objective(data).numpy()
            logf.append(elbo)
            nlpd.append(-tf.reduce_mean(model.predict_log_density((xt,yt))).numpy())

            # for var in optimizer.variables():
            #     var.assign(tf.zeros_like(var))

            for i in range(n_m_steps):
                optimization_step(model,training_loss,model.trainable_variables)

        else:

            # for var in optimizer2.variables():
            #     var.assign(tf.zeros_like(var))

            for i in range(n_e_steps):
                variational_params =  model.q_mu.trainable_variables + model.q_sqrt.trainable_variables
                #print(model.trainable_variables)
                optimization_step2(model,training_loss,variational_params)
                #elbo = model.maximum_log_likelihood_objective(data).numpy()
                #nl = -tf.reduce_mean(model.predict_log_density((xt,yt))).numpy()
            elbo = model.maximum_log_likelihood_objective(data).numpy()
            logf.append(elbo)
            nlpd.append(-tf.reduce_mean(model.predict_log_density((xt,yt))).numpy())
                #logf.append(elbo)
                #print(logf)
                #nlpd.append(nl)

            # for var in optimizer.variables():
            #     var.assign(tf.zeros_like(var))


            for i in range(n_m_steps):
                trainable_variables = model.kernel.trainable_variables + \
                                      model.likelihood.trainable_variables + \
                                      model.inducing_variable.trainable_variables

                optimization_step(model,training_loss,trainable_variables)



        # elbo = model.maximum_log_likelihood_objective(data).numpy()
        # logf.append(elbo)
        #
        # nlpd.append(-tf.reduce_mean(model.predict_log_density((xt,yt))).numpy())

        #if step % 10 == 0:
            #print(elbo)

    return logf,nlpd

ell = 1.0
var = 1.0

if data_name == 'elevators':
    # Load all the data
    data = np.array(loadmat('../../demos/data/elevators.mat')['data'])
    X = data[:, :-1]
    Y = data[:, -1].reshape(-1,1)
else:
    data,test = data_load(data_name,split=1.0,normalize=False)
    X,Y = data


X_scaler = StandardScaler().fit(X)
Y_scaler = StandardScaler().fit(Y)
X = X_scaler.transform(X)
Y = Y_scaler.transform(Y)
N = X.shape[0]
D = X.shape[1]

# Initialize inducing locations to the first M inputs in the dataset

#kmeans = KMeans(n_clusters=M, random_state=0).fit(X)
#Z = kmeans.cluster_centers_
Z = X[:M, :].copy()

kf = KFold(n_splits=n_folds, random_state=0, shuffle=True)


RMSE = np.zeros((nm,n_folds))
ERRP = np.zeros((nm,n_folds))
NLPD = np.zeros((nm,n_folds))
TIME = np.zeros((nm,n_folds))


NLPD_i = np.zeros((nm,nit,n_folds))
LOGF_i = np.zeros((nm,nit,n_folds))

fold = 0
for train_index, test_index in kf.split(X):

    # The data split
    x = X[train_index]
    y = Y[train_index]
    xt = X[test_index]
    yt = Y[test_index]

    if mb_size == 'full':
        mb_size = x.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat().shuffle(x.shape[0])

    mods,names = init_model(x.shape[0])

    maxiter = ci_niter(nit)

    j=0

    for m in mods:
        t0 = time.time()
        logf_i,nlpd_i = run_optim(m, maxiter)
        t = time.time()-t0

        nlpd = -tf.reduce_mean(m.predict_log_density((xt,yt))).numpy()
        Eft,_ = m.predict_f(xt,full_output_cov=False)
        rmse = tf.math.sqrt(tf.reduce_mean((yt-Eft)**2))

        yp,_ = m.predict_y(xt)
        errp = 1.-np.sum((yp>0.5)==(yt>0.5))/yt.shape[0]


        print('NLPD for {}: {}'.format(m.name,nlpd))
        print('ERR% for {}: {}'.format(m.name,rmse))

        # Store results
        ERRP[j,fold]=rmse
        NLPD[j,fold]=nlpd
        TIME[j,fold]=t

        NLPD_i[j,:,fold] = np.array(nlpd_i)
        LOGF_i[j,:,fold] = np.array(logf_i)

        j+=1


    fold+=1


# Calculate averages and standard deviations
rmse_mean = np.mean(ERRP,1)
rmse_std = np.std(ERRP,1)
nlpd_mean = np.mean(NLPD,1)
nlpd_std = np.std(NLPD,1)
time_mean = np.mean(TIME,1)
time_std = np.std(TIME,1)

elbo_mean = np.mean(LOGF_i,2)
nlpd_i_mean = np.mean(NLPD_i,2)

plt.title('ELBO'+'_' + data_name)
plt.plot(range(nit),elbo_mean[0,:][:],label=names[0])
plt.plot(range(nit),elbo_mean[1,:][:],label=names[1])
plt.plot(range(nit),elbo_mean[2,:][:],label=names[2])
plt.plot(range(nit),elbo_mean[3,:][:],label=names[3])
plt.legend()
plt.show()

plt.title('NLPD'+'_' + data_name)
plt.plot(range(nit),nlpd_i_mean[0,:][:],label=names[0])
plt.plot(range(nit),nlpd_i_mean[1,:][:],label=names[1])
plt.plot(range(nit),nlpd_i_mean[2,:][:],label=names[2])
plt.plot(range(nit),nlpd_i_mean[3,:][:],label=names[3])
plt.legend()
plt.show()

# Report
print('Data: {}, n: {}, m: {}, steps: {}'.format(data_name,x.shape[0],mb_size,nit))
print('{:<14} {:^13}   {:^13}   '.format('Method','NLPD','RMSE'))

# Report
print('Data: {}, n: {}, m: {}, steps: {}'.format(data_name,x.shape[0],mb_size,nit))
print('{:<14} {:^13}   {:^13}   {:^13}'.format('Method','NLPD','RMSE','TIME'))
for i in range(len(mods)):
    print('{:<14} {:.3f}+/-{:.3f}   {:.3f}+/-{:.3f}   {:.3f}+/-{:.3f}'.format(names[i],
                                                         nlpd_mean[i],nlpd_std[i],
                                                         rmse_mean[i],rmse_std[i],
                                                         time_mean[i],time_std[i]))

# Save results
np.savez('res/{}-{}-{}-{}.npz'.format(data_name,M,mb_size,nit), data_name=data_name, names=names,
         minibatch_size=mb_size, maxit=nit, NLPD=NLPD, RMSE=RMSE, TIME=TIME, N=N, D=D,I=M)

# Save results
np.savez('res/{}-{}-{}-{}-{}-{}-{}-{}.npz'.format(data_name,M,nit,optim,n_e_steps,n_m_steps,nat_lr,adam_lr),
         data_name=data_name, names=names,
         minibatch_size=mb_size, maxit=nit, NLPD=NLPD_i, ELBO=LOGF_i, N=N, D=D,I=M)