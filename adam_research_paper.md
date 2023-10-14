## ADAM: A METHOD FOR STOCHASTICOPTIMIZATION

```
Diederik P. Kingma*
University of Amsterdam, OpenAI
dpkingma@openai.com
```
```
Jimmy Lei Ba∗
University of Toronto
jimmy@psi.utoronto.ca
```
## ABSTRACT

```
We introduceAdam, an algorithm for first-order gradient-based optimization of
stochastic objective functions, based on adaptive estimates of lower-order mo-
ments. The method is straightforward to implement, is computationally efficient,
has little memory requirements, is invariant to diagonal rescaling of the gradients,
and is well suited for problems that are large in terms of data and/or parameters.
The method is also appropriate for non-stationary objectives and problems with
very noisy and/or sparse gradients. The hyper-parameters have intuitive interpre-
tations and typically require little tuning. Some connections to related algorithms,
on whichAdamwas inspired, are discussed. We also analyze the theoretical con-
vergence properties of the algorithm and provide a regret bound on the conver-
gence rate that is comparable to the best known results under the online convex
optimization framework. Empirical results demonstrate that Adam works well in
practice and compares favorably to other stochastic optimization methods. Finally,
we discussAdaMax, a variant ofAdambased on the infinity norm.
```
## 1 INTRODUCTION

```
Stochastic gradient-based optimization is of core practical importance in many fields of science and
engineering. Many problems in these fields can be cast as the optimization of some scalar parameter-
ized objective function requiring maximization or minimization with respect to its parameters. If the
function is differentiable w.r.t. its parameters, gradient descent is a relatively efficient optimization
method, since the computation of first-order partial derivatives w.r.t. all the parameters is of the same
computational complexity as just evaluating the function. Often, objective functions are stochastic.
For example, many objective functions are composed of a sum of subfunctions evaluated at different
subsamples of data; in this case optimization can be made more efficient by taking gradient steps
w.r.t. individual subfunctions, i.e. stochastic gradient descent (SGD) or ascent. SGD proved itself
as an efficient and effective optimization method that was central in many machine learning success
stories, such as recent advances in deep learning (Deng et al., 2013; Krizhevsky et al., 2012; Hinton
& Salakhutdinov, 2006; Hinton et al., 2012a; Graves et al., 2013). Objectives may also have other
sources of noise than data subsampling, such as dropout (Hinton et al., 2012b) regularization. For
all such noisy objectives, efficient stochastic optimization techniques are required. The focus of this
paper is on the optimization of stochastic objectives with high-dimensional parameters spaces. In
these cases, higher-order optimization methods are ill-suited, and discussion in this paper will be
restricted to first-order methods.
We proposeAdam, a method for efficient stochastic optimization that only requires first-order gra-
dients with little memory requirement. The method computes individual adaptive learning rates for
different parameters from estimates of first and second moments of the gradients; the nameAdam
is derived from adaptive moment estimation. Our method is designed to combine the advantages
of two recently popular methods: AdaGrad (Duchi et al., 2011), which works well with sparse gra-
dients, and RMSProp (Tieleman & Hinton, 2012), which works well in on-line and non-stationary
settings; important connections to these and other stochastic optimization methods are clarified in
section 5. Some of Adam’s advantages are that the magnitudes of parameter updates are invariant to
rescaling of the gradient, its stepsizes are approximately bounded by the stepsize hyperparameter,
it does not require a stationary objective, it works with sparse gradients, and it naturally performs a
form of step size annealing.
∗Equal contribution. Author ordering determined by coin flip over a Google Hangout.
```
# arXiv:1412.6980v9 [cs.LG] 30 Jan 2017


Algorithm 1:Adam, our proposed algorithm for stochastic optimization. See section 2 for details,
and for a slightly more efficient (but less clear) order of computation.g^2 tindicates the elementwise
squaregt gt. Good default settings for the tested machine learning problems areα= 0. 001 ,
β 1 = 0. 9 ,β 2 = 0. 999 and= 10−^8. All operations on vectors are element-wise. Withβt 1 andβt 2
we denoteβ 1 andβ 2 to the powert.

Require:α: Stepsize
Require:β 1 ,β 2 ∈[0,1): Exponential decay rates for the moment estimates
Require:f(θ): Stochastic objective function with parametersθ
Require:θ 0 : Initial parameter vector
m 0 ← 0 (Initialize 1stmoment vector)
v 0 ← 0 (Initialize 2ndmoment vector)
t← 0 (Initialize timestep)
whileθtnot convergeddo
t←t+ 1
gt←∇θft(θt− 1 )(Get gradients w.r.t. stochastic objective at timestept)
mt←β 1 ·mt− 1 + (1−β 1 )·gt(Update biased first moment estimate)
vt←β 2 ·vt− 1 + (1−β 2 )·g^2 t(Update biased second raw moment estimate)
m̂t←mt/(1−β 1 t)(Compute bias-corrected first moment estimate)
̂vt←vt/(1−β 2 t)(Compute bias-corrected second raw moment estimate)
θt←θt− 1 −α·m̂t/(

### √

```
̂vt+)(Update parameters)
end while
returnθt(Resulting parameters)
```
In section 2 we describe the algorithm and the properties of its update rule. Section 3 explains
our initialization bias correction technique, and section 4 provides a theoretical analysis of Adam’s
convergence in online convex programming. Empirically, our method consistently outperforms other
methods for a variety of models and datasets, as shown in section 6. Overall, we show that Adam is
a versatile algorithm that scales to large-scale high-dimensional machine learning problems.

## 2 ALGORITHM

See algorithm 1 for pseudo-code of our proposed algorithmAdam. Letf(θ)be a noisy objec-
tive function: a stochastic scalar function that is differentiable w.r.t. parametersθ. We are in-
terested in minimizing the expected value of this function,E[f(θ)]w.r.t. its parametersθ. With
f 1 (θ),...,,fT(θ)we denote the realisations of the stochastic function at subsequent timesteps
1 ,...,T. The stochasticity might come from the evaluation at random subsamples (minibatches)
of datapoints, or arise from inherent function noise. Withgt=∇θft(θ)we denote the gradient, i.e.
the vector of partial derivatives offt, w.r.tθevaluated at timestept.

The algorithm updates exponential moving averages of the gradient (mt) and the squared gradient
(vt) where the hyper-parametersβ 1 ,β 2 ∈[0,1)control the exponential decay rates of these moving
averages. The moving averages themselves are estimates of the 1stmoment (the mean) and the
2 ndraw moment (the uncentered variance) of the gradient. However, these moving averages are
initialized as (vectors of) 0’s, leading to moment estimates that are biased towards zero, especially
during the initial timesteps, and especially when the decay rates are small (i.e. theβs are close to 1).
The good news is that this initialization bias can be easily counteracted, resulting in bias-corrected
estimatesm̂tand̂vt. See section 3 for more details.

Note that the efficiency of algorithm 1 can, at the expense of clarity, be improved upon by changing
the order of computation, e.g. by replacing the last three lines in the loop with the following lines:
αt=α·

### √

```
1 −βt 2 /(1−βt 1 )andθt←θt− 1 −αt·mt/(
```
### √

```
vt+ ˆ).
```
### 2.1 ADAM’S UPDATE RULE

An important property of Adam’s update rule is its careful choice of stepsizes. Assuming= 0, the
effective step taken in parameter space at timesteptis∆t=α·m̂t/

### √

̂vt. The effective stepsize has
two upper bounds:|∆t| ≤α·(1−β 1 )/

### √

```
1 −β 2 in the case(1−β 1 )>
```
### √

```
1 −β 2 , and|∆t| ≤α
```

otherwise. The first case only happens in the most severe case of sparsity: when a gradient has
been zero at all timesteps except at the current timestep. For less sparse cases, the effective stepsize
will be smaller. When(1−β 1 ) =

### √

```
1 −β 2 we have that|m̂t/
```
### √

```
̂vt|< 1 therefore|∆t|< α. In
```
more common scenarios, we will have thatm̂t/

### √

```
̂vt≈± 1 since|E[g]/
```
### √

E[g^2 ]|≤ 1. The effective
magnitude of the steps taken in parameter space at each timestep are approximately bounded by
the stepsize settingα, i.e.,|∆t|/α. This can be understood as establishing atrust regionaround
the current parameter value, beyond which the current gradient estimate does not provide sufficient
information. This typically makes it relatively easy to know the right scale ofαin advance. For
many machine learning models, for instance, we often know in advance that good optima are with
high probability within some set region in parameter space; it is not uncommon, for example, to
have a prior distribution over the parameters. Sinceαsets (an upper bound of) the magnitude of
steps in parameter space, we can often deduce the right order of magnitude ofαsuch that optima
can be reached fromθ 0 within some number of iterations. With a slight abuse of terminology,
we will call the ratiom̂t/

### √

̂vtthesignal-to-noiseratio (SNR). With a smaller SNR the effective
stepsize∆twill be closer to zero. This is a desirable property, since a smaller SNR means that
there is greater uncertainty about whether the direction ofm̂tcorresponds to the direction of the true
gradient. For example, the SNR value typically becomes closer to 0 towards an optimum, leading
to smaller effective steps in parameter space: a form of automatic annealing. The effective stepsize
∆tis also invariant to the scale of the gradients; rescaling the gradientsgwith factorcwill scalem̂t
with a factorcand̂vtwith a factorc^2 , which cancel out:(c·m̂t)/(

### √

```
c^2 ·̂vt) =m̂t/
```
### √

```
̂vt.
```
## 3 INITIALIZATION BIAS CORRECTION

As explained in section 2, Adam utilizes initialization bias correction terms. We will here derive
the term for the second moment estimate; the derivation for the first moment estimate is completely
analogous. Letgbe the gradient of the stochastic objectivef, and we wish to estimate its second
raw moment (uncentered variance) using an exponential moving average of the squared gradient,
with decay rateβ 2. Letg 1 ,...,gTbe the gradients at subsequent timesteps, each a draw from an
underlying gradient distributiongt∼p(gt). Let us initialize the exponential moving average as
v 0 = 0(a vector of zeros). First note that the update at timesteptof the exponential moving average
vt=β 2 ·vt− 1 + (1−β 2 )·gt^2 (wheregt^2 indicates the elementwise squaregt gt) can be written as
a function of the gradients at all previous timesteps:

```
vt= (1−β 2 )
```
```
∑t
```
```
i=
```
```
β 2 t−i·gi^2 (1)
```
We wish to know howE[vt], the expected value of the exponential moving average at timestept,
relates to the true second momentE[g^2 t], so we can correct for the discrepancy between the two.
Taking expectations of the left-hand and right-hand sides of eq. (1):

```
E[vt] =E
```
### [

```
(1−β 2 )
```
```
∑t
```
```
i=
```
```
β 2 t−i·g^2 i
```
### ]

### (2)

```
=E[g^2 t]·(1−β 2 )
```
```
∑t
```
```
i=
```
```
βt 2 −i+ζ (3)
```
```
=E[g^2 t]·(1−βt 2 ) +ζ (4)
```
whereζ= 0if the true second momentE[g^2 i]is stationary; otherwiseζcan be kept small since
the exponential decay rateβ 1 can (and should) be chosen such that the exponential moving average
assigns small weights to gradients too far in the past. What is left is the term(1−βt 2 )which is
caused by initializing the running average with zeros. In algorithm 1 we therefore divide by this
term to correct the initialization bias.

In case of sparse gradients, for a reliable estimate of the second moment one needs to average over
many gradients by chosing a small value ofβ 2 ; however it is exactly this case of smallβ 2 where a
lack of initialisation bias correction would lead to initial steps that are much larger.


## 4 CONVERGENCE ANALYSIS

We analyze the convergence of Adam using the online learning framework proposed in (Zinkevich,
2003). Given an arbitrary, unknown sequence of convex cost functionsf 1 (θ),f 2 (θ),...,fT(θ). At
each timet, our goal is to predict the parameterθtand evaluate it on a previously unknown cost
functionft. Since the nature of the sequence is unknown in advance, we evaluate our algorithm
using the regret, that is the sum of all the previous difference between the online predictionft(θt)
and the best fixed point parameterft(θ∗)from a feasible setXfor all the previous steps. Concretely,
the regret is defined as:

### R(T) =

### ∑T

```
t=
```
```
[ft(θt)−ft(θ∗)] (5)
```
whereθ∗= arg minθ∈X

### ∑T

```
t=1ft(θ). We show Adam hasO(
```
### √

T)regret bound and a proof is given
in the appendix. Our result is comparable to the best known bound for this general convex online
learning problem. We also use some definitions simplify our notation, wheregt,∇ft(θt)andgt,i
as theithelement. We defineg1:t,i∈Rtas a vector that contains theithdimension of the gradients

over all iterations tillt,g1:t,i= [g 1 ,i,g 2 ,i,···,gt,i]. Also, we defineγ, β

(^21)
√
β 2. Our following
theorem holds when the learning rateαtis decaying at a rate oft−
(^12)
and first moment running
average coefficientβ 1 ,tdecay exponentially withλ, that is typically close to 1, e.g. 1 − 10 −^8.
Theorem 4.1.Assume that the functionfthas bounded gradients,‖∇ft(θ)‖ 2 ≤G,‖∇ft(θ)‖∞≤
G∞for allθ∈Rdand distance between anyθtgenerated by Adam is bounded,‖θn−θm‖ 2 ≤D,
‖θm−θn‖∞≤D∞for anym,n∈ { 1 ,...,T}, andβ 1 ,β 2 ∈[0,1)satisfy β
(^21)
√β
2 <^1. Letαt=
√α
t
andβ 1 ,t=β 1 λt−^1 ,λ∈(0,1). Adam achieves the following guarantee, for allT≥ 1.

### R(T)≤

### D^2

```
2 α(1−β 1 )
```
```
∑d
```
```
i=
```
### √

```
T̂vT,i+
```
```
α(1 +β 1 )G∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2 +
```
```
∑d
```
```
i=
```
### D^2 ∞G∞

### √

```
1 −β 2
2 α(1−β 1 )(1−λ)^2
```
Our Theorem 4.1 implies when the data features are sparse and bounded gradients, the sum-

mation term can be much smaller than its upper bound

```
∑d
i=1‖g1:T,i‖^2 << dG∞
```
### √

T and
∑d
i=

### √

```
T̂vT,i<< dG∞
```
### √

```
T, in particular if the class of function and data features are in the form of
```
section 1.2 in (Duchi et al., 2011). Their results for the expected valueE[

∑d
i=1‖g1:T,i‖^2 ]also apply
to Adam. In particular, the adaptive method, such as Adam and Adagrad, can achieveO(logd

### √

### T),

an improvement overO(

### √

dT)for the non-adaptive method. Decayingβ 1 ,ttowards zero is impor-
tant in our theoretical analysis and also matches previous empirical findings, e.g. (Sutskever et al.,
2013) suggests reducing the momentum coefficient in the end of training can improve convergence.

Finally, we can show the average regret of Adam converges,

Corollary 4.2.Assume that the functionfthas bounded gradients,‖∇ft(θ)‖ 2 ≤G,‖∇ft(θ)‖∞≤
G∞for allθ∈Rdand distance between anyθtgenerated by Adam is bounded,‖θn−θm‖ 2 ≤D,
‖θm−θn‖∞≤D∞for anym,n∈ { 1 ,...,T}. Adam achieves the following guarantee, for all
T≥ 1.
R(T)
T

### =O(

### 1

### √

### T

### )

This result can be obtained by using Theorem 4.1 and

```
∑d
i=1‖g1:T,i‖^2 ≤ dG∞
```
### √

```
T. Thus,
```
limT→∞R(TT)= 0.

## 5 RELATED WORK

Optimization methods bearing a direct relation to Adam are RMSProp (Tieleman & Hinton, 2012;
Graves, 2013) and AdaGrad (Duchi et al., 2011); these relationships are discussed below. Other
stochastic optimization methods include vSGD (Schaul et al., 2012), AdaDelta (Zeiler, 2012) and the
natural Newton method from Roux & Fitzgibbon (2010), all setting stepsizes by estimating curvature


from first-order information. The Sum-of-Functions Optimizer (SFO) (Sohl-Dickstein et al., 2014)
is a quasi-Newton method based on minibatches, but (unlike Adam) has memory requirements linear
in the number of minibatch partitions of a dataset, which is often infeasible on memory-constrained
systems such as a GPU. Like natural gradient descent (NGD) (Amari, 1998), Adam employs a
preconditioner that adapts to the geometry of the data, sincêvtis an approximation to the diagonal
of the Fisher information matrix (Pascanu & Bengio, 2013); however, Adam’s preconditioner (like
AdaGrad’s) is more conservative in its adaption than vanilla NGD by preconditioning with the square
root of the inverse of the diagonal Fisher information matrix approximation.

RMSProp: An optimization method closely related to Adam is RMSProp (Tieleman & Hinton,
2012). A version with momentum has sometimes been used (Graves, 2013). There are a few impor-
tant differences between RMSProp with momentum and Adam: RMSProp with momentum gener-
ates its parameter updates using a momentum on the rescaled gradient, whereas Adam updates are
directly estimated using a running average of first and second moment of the gradient. RMSProp
also lacks a bias-correction term; this matters most in case of a value ofβ 2 close to 1 (required in
case of sparse gradients), since in that case not correcting the bias leads to very large stepsizes and
often divergence, as we also empirically demonstrate in section 6.4.

AdaGrad: An algorithm that works well for sparse gradients is AdaGrad (Duchi et al., 2011). Its

basic version updates parameters asθt+1=θt−α·gt/

### √∑

```
t
i=1g
```
```
2
t. Note that if we chooseβ^2 to be
```
infinitesimally close to 1 from below, thenlimβ 2 → 1 ̂vt=t−^1 ·

```
∑t
i=1g
```
2
t. AdaGrad corresponds to a
version of Adam withβ 1 = 0, infinitesimal(1−β 2 )and a replacement ofαby an annealed version

αt=α·t−^1 /^2 , namelyθt−α·t−^1 /^2 ·m̂t/

### √

```
limβ 2 → 1 ̂vt=θt−α·t−^1 /^2 ·gt/
```
### √

```
t−^1 ·
```
```
∑t
i=1g
```
```
2
t=
```
θt−α·gt/

### √∑

```
t
i=1g
```
2
t. Note that this direct correspondence between Adam and Adagrad does
not hold when removing the bias-correction terms; without bias correction, like in RMSProp, aβ 2
infinitesimally close to 1 would lead to infinitely large bias, and infinitely large parameter updates.

## 6 EXPERIMENTS

To empirically evaluate the proposed method, we investigated different popular machine learning
models, including logistic regression, multilayer fully connected neural networks and deep convolu-
tional neural networks. Using large models and datasets, we demonstrate Adam can efficiently solve
practical deep learning problems.

We use the same parameter initialization when comparing different optimization algorithms. The
hyper-parameters, such as learning rate and momentum, are searched over a dense grid and the
results are reported using the best hyper-parameter setting.

### 6.1 EXPERIMENT: LOGISTICREGRESSION

We evaluate our proposed method on L2-regularized multi-class logistic regression using the MNIST
dataset. Logistic regression has a well-studied convex objective, making it suitable for comparison
of different optimizers without worrying about local minimum issues. The stepsizeαin our logistic
regression experiments is adjusted by 1 /

### √

tdecay, namelyαt=√αtthat matches with our theorat-
ical prediction from section 4. The logistic regression classifies the class label directly on the 784
dimension image vectors. We compare Adam to accelerated SGD with Nesterov momentum and
Adagrad using minibatch size of 128. According to Figure 1, we found that the Adam yields similar
convergence as SGD with momentum and both converge faster than Adagrad.

As discussed in (Duchi et al., 2011), Adagrad can efficiently deal with sparse features and gradi-
ents as one of its main theoretical results whereas SGD is low at learning rare features. Adam with
1 /

### √

tdecay on its stepsize should theoratically match the performance of Adagrad. We examine the
sparse feature problem using IMDB movie review dataset from (Maas et al., 2011). We pre-process
the IMDB movie reviews into bag-of-words (BoW) feature vectors including the first 10,000 most
frequent words. The 10,000 dimension BoW feature vector for each review is highly sparse. As sug-
gested in (Wang & Manning, 2013), 50% dropout noise can be applied to the BoW features during


```
0 5 10 15 20 25 30 35 40 45
iterations over entire dataset
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
training cost
```
```
MNIST Logistic Regression
AdaGrad
SGDNesterov
Adam
```
```
0 20 40 60 80 100 120 140 160
iterations over entire dataset
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
0.
```
```
training cost
```
```
IMDB BoW feature Logistic Regression
Adagrad+dropout
RMSProp+dropout
SGDNesterov+dropout
Adam+dropout
```
Figure 1: Logistic regression training negative log likelihood on MNIST images and IMDB movie
reviews with 10,000 bag-of-words (BoW) feature vectors.

training to prevent over-fitting. In figure 1, Adagrad outperforms SGD with Nesterov momentum
by a large margin both with and without dropout noise. Adam converges as fast as Adagrad. The
empirical performance of Adam is consistent with our theoretical findings in sections 2 and 4. Sim-
ilar to Adagrad, Adam can take advantage of sparse features and obtain faster convergence rate than
normal SGD with momentum.

### 6.2 EXPERIMENT: MULTI-LAYERNEURALNETWORKS

Multi-layer neural network are powerful models with non-convex objective functions. Although
our convergence analysis does not apply to non-convex problems, we empirically found that Adam
often outperforms other methods in such cases. In our experiments, we made model choices that are
consistent with previous publications in the area; a neural network model with two fully connected
hidden layers with 1000 hidden units each and ReLU activation are used for this experiment with
minibatch size of 128.

First, we study different optimizers using the standard deterministic cross-entropy objective func-
tion withL 2 weight decay on the parameters to prevent over-fitting. The sum-of-functions (SFO)
method (Sohl-Dickstein et al., 2014) is a recently proposed quasi-Newton method that works with
minibatches of data and has shown good performance on optimization of multi-layer neural net-
works. We used their implementation and compared with Adam to train such models. Figure 2
shows that Adam makes faster progress in terms of both the number of iterations and wall-clock
time. Due to the cost of updating curvature information, SFO is 5-10x slower per iteration com-
pared to Adam, and has a memory requirement that is linear in the number minibatches.

Stochastic regularization methods, such as dropout, are an effective way to prevent over-fitting and
often used in practice due to their simplicity. SFO assumes deterministic subfunctions, and indeed
failed to converge on cost functions with stochastic regularization. We compare the effectiveness of
Adam to other stochastic first order methods on multi-layer neural networks trained with dropout
noise. Figure 2 shows our results; Adam shows better convergence than other methods.

### 6.3 EXPERIMENT: CONVOLUTIONALNEURALNETWORKS

Convolutional neural networks (CNNs) with several layers of convolution, pooling and non-linear
units have shown considerable success in computer vision tasks. Unlike most fully connected neural
nets, weight sharing in CNNs results in vastly different gradients in different layers. A smaller
learning rate for the convolution layers is often used in practice when applying SGD. We show the
effectiveness of Adam in deep CNNs. Our CNN architecture has three alternating stages of 5x
convolution filters and 3x3 max pooling with stride of 2 that are followed by a fully connected layer
of 1000 rectified linear hidden units (ReLU’s). The input image are pre-processed by whitening, and


```
0 50 100 150 200
iterations over entire dataset
```
```
10 -
```
```
10 -
```
```
training cost
```
```
MNIST Multilayer Neural Network + dropout
AdaGrad
RMSProp
SGDNesterov
AdaDelta
Adam
```
```
(a) (b)
```
Figure 2: Training of multilayer neural networks on MNIST images. (a) Neural networks using
dropout stochastic regularization. (b) Neural networks with deterministic cost function. We compare
with the sum-of-functions (SFO) optimizer (Sohl-Dickstein et al., 2014)

```
0.50.0 0.5 iterations over entire dataset1.0 1.5 2.0 2.5 3.
```
```
1.
```
```
1.
```
```
2.
```
```
2.
```
```
3.
```
```
training cost
```
```
CIFAR10 ConvNet First 3 Epoches
AdaGrad
AdaGrad+dropout
SGDNesterov
SGDNesterov+dropout
Adam
Adam+dropout
```
10 -4 (^0510) iterations over entire dataset 15 20 25 30 35 40 45
10 -
10 -
10 -
100
101
102
training cost
CIFAR10 ConvNet
AdaGrad
AdaGrad+dropout
SGDNesterov
SGDNesterov+dropout
Adam
Adam+dropout
Figure 3: Convolutional neural networks training cost. (left) Training cost for the first three epochs.
(right) Training cost over 45 epochs. CIFAR-10 with c64-c64-c128-1000 architecture.
dropout noise is applied to the input layer and fully connected layer. The minibatch size is also set
to 128 similar to previous experiments.
Interestingly, although both Adam and Adagrad make rapid progress lowering the cost in the initial
stage of the training, shown in Figure 3 (left), Adam and SGD eventually converge considerably
faster than Adagrad for CNNs shown in Figure 3 (right). We notice the second moment estimatêvt
vanishes to zeros after a few epochs and is dominated by thein algorithm 1. The second moment
estimate is therefore a poor approximation to the geometry of the cost function in CNNs comparing
to fully connected network from Section 6.2. Whereas, reducing the minibatch variance through
the first moment is more important in CNNs and contributes to the speed-up. As a result, Adagrad
converges much slower than others in this particular experiment. Though Adam shows marginal
improvement over SGD with momentum, it adapts learning rate scale for different layers instead of
hand picking manually as in SGD.


```
β 1 = 0
```
```
β 1 =0. 9
```
```
β 2 =0. 99 β 2 =0. 999 β 2 =0. 9999 β 2 =0. 99 β 2 =0. 999 β 2 =0. 9999
```
```
(a) after 10 epochs (b) after 100 epochs
```
```
log 10 (α)
```
```
Loss
```
Figure 4: Effect of bias-correction terms (red line) versus no bias correction terms (green line)
after 10 epochs (left) and 100 epochs (right) on the loss (y-axes) when learning a Variational Auto-
Encoder (VAE) (Kingma & Welling, 2013), for different settings of stepsizeα(x-axes) and hyper-
parametersβ 1 andβ 2.

### 6.4 EXPERIMENT:BIAS-CORRECTION TERM

We also empirically evaluate the effect of the bias correction terms explained in sections 2 and 3.
Discussed in section 5, removal of the bias correction terms results in a version of RMSProp (Tiele-
man & Hinton, 2012) with momentum. We vary theβ 1 andβ 2 when training a variational auto-
encoder (VAE) with the same architecture as in (Kingma & Welling, 2013) with a single hidden
layer with 500 hidden units with softplus nonlinearities and a 50-dimensional spherical Gaussian
latent variable. We iterated over a broad range of hyper-parameter choices, i.e.β 1 ∈[0, 0 .9]and
β 2 ∈[0. 99 , 0. 999 , 0 .9999], andlog 10 (α)∈[− 5 ,...,−1]. Values ofβ 2 close to 1, required for robust-
ness to sparse gradients, results in larger initialization bias; therefore we expect the bias correction
term is important in such cases of slow decay, preventing an adverse effect on optimization.

In Figure 4, valuesβ 2 close to 1 indeed lead to instabilities in training when no bias correction term
was present, especially at first few epochs of the training. The best results were achieved with small
values of(1−β 2 )and bias correction; this was more apparent towards the end of optimization when
gradients tends to become sparser as hidden units specialize to specific patterns. In summary, Adam
performed equal or better than RMSProp, regardless of hyper-parameter setting.

## 7 EXTENSIONS

### 7.1 ADAMAX

In Adam, the update rule for individual weights is to scale their gradients inversely proportional to a
(scaled)L^2 norm of their individual current and past gradients. We can generalize theL^2 norm based
update rule to aLpnorm based update rule. Such variants become numerically unstable for large
p. However, in the special case where we letp→ ∞, a surprisingly simple and stable algorithm
emerges; see algorithm 2. We’ll now derive the algorithm. Let, in case of theLpnorm, the stepsize

at timetbe inversely proportional tov^1 t/p, where:

```
vt=β 2 pvt− 1 + (1−β 2 p)|gt|p (6)
```
```
= (1−βp 2 )
```
```
∑t
```
```
i=
```
```
β 2 p(t−i)·|gi|p (7)
```

Algorithm 2:AdaMax, a variant of Adam based on the infinity norm. See section 7.1 for details.
Good default settings for the tested machine learning problems areα = 0. 002 ,β 1 = 0. 9 and
β 2 = 0. 999. Withβ 1 twe denoteβ 1 to the powert. Here,(α/(1−βt 1 ))is the learning rate with the
bias-correction term for the first moment. All operations on vectors are element-wise.

Require:α: Stepsize
Require:β 1 ,β 2 ∈[0,1): Exponential decay rates
Require:f(θ): Stochastic objective function with parametersθ
Require:θ 0 : Initial parameter vector
m 0 ← 0 (Initialize 1stmoment vector)
u 0 ← 0 (Initialize the exponentially weighted infinity norm)
t← 0 (Initialize timestep)
whileθtnot convergeddo
t←t+ 1
gt←∇θft(θt− 1 )(Get gradients w.r.t. stochastic objective at timestept)
mt←β 1 ·mt− 1 + (1−β 1 )·gt(Update biased first moment estimate)
ut←max(β 2 ·ut− 1 ,|gt|)(Update the exponentially weighted infinity norm)
θt←θt− 1 −(α/(1−βt 1 ))·mt/ut(Update parameters)
end while
returnθt(Resulting parameters)

Note that the decay term is here equivalently parameterised asβ 2 pinstead ofβ 2. Now letp→ ∞,
and defineut= limp→∞(vt)^1 /p, then:

```
ut= lim
p→∞
```
```
(vt)^1 /p= lim
p→∞
```
### (

```
(1−β 2 p)
```
```
∑t
```
```
i=
```
```
βp 2 (t−i)·|gi|p
```
```
) 1 /p
```
```
(8)
```
```
= lim
p→∞
```
```
(1−βp 2 )^1 /p
```
```
(t
∑
```
```
i=
```
```
β 2 p(t−i)·|gi|p
```
```
) 1 /p
```
```
(9)
```
```
= lim
p→∞
```
```
( t
∑
```
```
i=
```
### (

```
β 2 (t−i)·|gi|
```
```
)p
```
```
) 1 /p
(10)
```
```
= max
```
### (

```
β 2 t−^1 |g 1 |,β 2 t−^2 |g 2 |,...,β 2 |gt− 1 |,|gt|
```
### )

### (11)

Which corresponds to the remarkably simple recursive formula:

```
ut= max(β 2 ·ut− 1 ,|gt|) (12)
```
with initial valueu 0 = 0. Note that, conveniently enough, we don’t need to correct for initialization
bias in this case. Also note that the magnitude of parameter updates has a simpler bound with
AdaMax than Adam, namely:|∆t|≤α.

### 7.2 TEMPORAL AVERAGING

Since the last iterate is noisy due to stochastic approximation, better generalization performance is
often achieved by averaging. Previously in Moulines & Bach (2011), Polyak-Ruppert averaging
(Polyak & Juditsky, 1992; Ruppert, 1988) has been shown to improve the convergence of standard
SGD, whereθ ̄t=^1 t

∑n
k=1θk. Alternatively, an exponential moving average over the parameters can
be used, giving higher weight to more recent parameter values. This can be trivially implemented
by adding one line to the inner loop of algorithms 1 and 2:θ ̄t←β 2 ·θ ̄t− 1 + (1−β 2 )θt, withθ ̄ 0 = 0.

Initalization bias can again be corrected by the estimatorθ̂t=θ ̄t/(1−β 2 t).

## 8 CONCLUSION

We have introduced a simple and computationally efficient algorithm for gradient-based optimiza-
tion of stochastic objective functions. Our method is aimed towards machine learning problems with


large datasets and/or high-dimensional parameter spaces. The method combines the advantages of
two recently popular optimization methods: the ability of AdaGrad to deal with sparse gradients,
and the ability of RMSProp to deal with non-stationary objectives. The method is straightforward
to implement and requires little memory. The experiments confirm the analysis on the rate of con-
vergence in convex problems. Overall, we found Adam to be robust and well-suited to a wide range
of non-convex optimization problems in the field machine learning.

## 9 ACKNOWLEDGMENTS

This paper would probably not have existed without the support of Google Deepmind. We would
like to give special thanks to Ivo Danihelka, and Tom Schaul for coining the name Adam. Thanks to
Kai Fan from Duke University for spotting an error in the original AdaMax derivation. Experiments
in this work were partly carried out on the Dutch national e-infrastructure with the support of SURF
Foundation. Diederik Kingma is supported by the Google European Doctorate Fellowship in Deep
Learning.

## REFERENCES

Amari, Shun-Ichi. Natural gradient works efficiently in learning.Neural computation, 10(2):251–276, 1998.

Deng, Li, Li, Jinyu, Huang, Jui-Ting, Yao, Kaisheng, Yu, Dong, Seide, Frank, Seltzer, Michael, Zweig, Geoff,
He, Xiaodong, Williams, Jason, et al. Recent advances in deep learning for speech research at microsoft.
ICASSP 2013, 2013.

Duchi, John, Hazan, Elad, and Singer, Yoram. Adaptive subgradient methods for online learning and stochastic
optimization.The Journal of Machine Learning Research, 12:2121–2159, 2011.

Graves, Alex. Generating sequences with recurrent neural networks.arXiv preprint arXiv:1308.0850, 2013.

Graves, Alex, Mohamed, Abdel-rahman, and Hinton, Geoffrey. Speech recognition with deep recurrent neural
networks. InAcoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on,
pp. 6645–6649. IEEE, 2013.

Hinton, G.E. and Salakhutdinov, R.R. Reducing the dimensionality of data with neural networks.Science, 313
(5786):504–507, 2006.

Hinton, Geoffrey, Deng, Li, Yu, Dong, Dahl, George E, Mohamed, Abdel-rahman, Jaitly, Navdeep, Senior,
Andrew, Vanhoucke, Vincent, Nguyen, Patrick, Sainath, Tara N, et al. Deep neural networks for acoustic
modeling in speech recognition: The shared views of four research groups.Signal Processing Magazine,
IEEE, 29(6):82–97, 2012a.

Hinton, Geoffrey E, Srivastava, Nitish, Krizhevsky, Alex, Sutskever, Ilya, and Salakhutdinov, Ruslan R. Im-
proving neural networks by preventing co-adaptation of feature detectors.arXiv preprint arXiv:1207.0580,
2012b.

Kingma, Diederik P and Welling, Max. Auto-Encoding Variational Bayes. InThe 2nd International Conference
on Learning Representations (ICLR), 2013.

Krizhevsky, Alex, Sutskever, Ilya, and Hinton, Geoffrey E. Imagenet classification with deep convolutional
neural networks. InAdvances in neural information processing systems, pp. 1097–1105, 2012.

Maas, Andrew L, Daly, Raymond E, Pham, Peter T, Huang, Dan, Ng, Andrew Y, and Potts, Christopher.
Learning word vectors for sentiment analysis. InProceedings of the 49th Annual Meeting of the Association
for Computational Linguistics: Human Language Technologies-Volume 1, pp. 142–150. Association for
Computational Linguistics, 2011.

Moulines, Eric and Bach, Francis R. Non-asymptotic analysis of stochastic approximation algorithms for
machine learning. InAdvances in Neural Information Processing Systems, pp. 451–459, 2011.

Pascanu, Razvan and Bengio, Yoshua. Revisiting natural gradient for deep networks. arXiv preprint
arXiv:1301.3584, 2013.

Polyak, Boris T and Juditsky, Anatoli B. Acceleration of stochastic approximation by averaging.SIAM Journal
on Control and Optimization, 30(4):838–855, 1992.


Roux, Nicolas L and Fitzgibbon, Andrew W. A fast natural newton method. InProceedings of the 27th
International Conference on Machine Learning (ICML-10), pp. 623–630, 2010.

Ruppert, David. Efficient estimations from a slowly convergent robbins-monro process. Technical report,
Cornell University Operations Research and Industrial Engineering, 1988.

Schaul, Tom, Zhang, Sixin, and LeCun, Yann. No more pesky learning rates.arXiv preprint arXiv:1206.1106,
2012.

Sohl-Dickstein, Jascha, Poole, Ben, and Ganguli, Surya. Fast large-scale optimization by unifying stochas-
tic gradient and quasi-newton methods. InProceedings of the 31st International Conference on Machine
Learning (ICML-14), pp. 604–612, 2014.

Sutskever, Ilya, Martens, James, Dahl, George, and Hinton, Geoffrey. On the importance of initialization and
momentum in deep learning. InProceedings of the 30th International Conference on Machine Learning
(ICML-13), pp. 1139–1147, 2013.

Tieleman, T. and Hinton, G. Lecture 6.5 - RMSProp, COURSERA: Neural Networks for Machine Learning.
Technical report, 2012.

Wang, Sida and Manning, Christopher. Fast dropout training. InProceedings of the 30th International Confer-
ence on Machine Learning (ICML-13), pp. 118–126, 2013.

Zeiler, Matthew D. Adadelta: An adaptive learning rate method.arXiv preprint arXiv:1212.5701, 2012.

Zinkevich, Martin. Online convex programming and generalized infinitesimal gradient ascent. 2003.


## 10 APPENDIX

### 10.1 CONVERGENCEPROOF

Definition 10.1.A functionf:Rd→Ris convex if for allx,y∈Rd, for allλ∈[0,1],

```
λf(x) + (1−λ)f(y)≥f(λx+ (1−λ)y)
```
Also, notice that a convex function can be lower bounded by a hyperplane at its tangent.

Lemma 10.2.If a functionf:Rd→Ris convex, then for allx,y∈Rd,

```
f(y)≥f(x) +∇f(x)T(y−x)
```
The above lemma can be used to upper bound the regret and our proof for the main theorem is
constructed by substituting the hyperplane with the Adam update rules.

The following two lemmas are used to support our main theorem. We also use some definitions sim-
plify our notation, wheregt,∇ft(θt)andgt,ias theithelement. We defineg1:t,i∈Rtas a vector
that contains theithdimension of the gradients over all iterations tillt,g1:t,i= [g 1 ,i,g 2 ,i,···,gt,i]

Lemma 10.3.Letgt=∇ft(θt)andg1:tbe defined as above and bounded,‖gt‖ 2 ≤G,‖gt‖∞≤
G∞. Then,

```
∑T
```
```
t=
```
### √

```
g^2 t,i
t
```
```
≤ 2 G∞‖g1:T,i‖ 2
```
Proof.We will prove the inequality using induction over T.

The base case forT= 1, we have

### √

```
g^21 ,i≤ 2 G∞‖g 1 ,i‖ 2.
```
For the inductive step,

```
∑T
```
```
t=
```
### √

```
g^2 t,i
t
```
### =

### T∑− 1

```
t=
```
### √

```
gt,i^2
t
```
### +

### √

```
gT,i^2
T
```
```
≤ 2 G∞‖g1:T− 1 ,i‖ 2 +
```
### √

```
gT,i^2
T
```
### = 2G∞

### √

```
‖g1:T,i‖^22 −g^2 T+
```
### √

```
gT,i^2
T
```
From,‖g1:T,i‖^22 −gT,i^2 +

```
gT,i^4
4 ‖g1:T,i‖^22 ≥ ‖g1:T,i‖
```
```
2
2 −g
```
2
T,i, we can take square root of both side and
have,

```
√
‖g1:T,i‖^22 −g^2 T,i≤‖g1:T,i‖ 2 −
```
```
g^2 T,i
2 ‖g1:T,i‖ 2
```
```
≤‖g1:T,i‖ 2 −
```
```
gT,i^2
2
```
### √

### TG^2 ∞

Rearrange the inequality and substitute the

### √

```
‖g1:T,i‖^22 −g^2 T,iterm,
```
### G∞

### √

```
‖g1:T,i‖^22 −gT^2 +
```
### √

```
g^2 T,i
T
```
```
≤ 2 G∞‖g1:T,i‖ 2
```

Lemma 10.4.Letγ, β

(^21)
√β
2. Forβ^1 ,β^2 ∈[0,1)that satisfy
√β^21
β 2 <^1 and boundedgt,‖gt‖^2 ≤G,
‖gt‖∞≤G∞, the following inequality holds
∑T
t=
m̂^2 t,i
√
t̂vt,i

### ≤

### 2

```
1 −γ
```
### 1

### √

```
1 −β 2
```
```
‖g1:T,i‖ 2
```
Proof.Under the assumption,

### √

```
1 −βt 2
(1−βt 1 )^2 ≤
```
1
(1−β 1 )^2. We can expand the last term in the summation
using the update rules in Algorithm 1,

```
∑T
```
```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### =

### T∑− 1

```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### +

### √

```
1 −β 2 T
(1−β 1 T)^2
```
### (

### ∑T

```
k=1(1−β^1 )β
```
```
T−k
1 gk,i)
```
```
2
√
T
```
### ∑T

```
j=1(1−β^2 )β
```
```
T−j
2 g
```
```
2
j,i
```
### ≤

### T∑− 1

```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### +

### √

```
1 −β 2 T
(1−β 1 T)^2
```
### ∑T

```
k=
```
```
T((1−β 1 )βT 1 −kgk,i)^2
√
T
```
### ∑T

```
j=1(1−β^2 )β
```
```
T−j
2 g
```
```
2
j,i
```
### ≤

### T∑− 1

```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### +

### √

```
1 −β 2 T
(1−β 1 T)^2
```
### ∑T

```
k=
```
```
T((1−β 1 )β 1 T−kgk,i)^2
√
T(1−β 2 )β 2 T−kg^2 k,i
```
### ≤

### T∑− 1

```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### +

### √

```
1 −β 2 T
(1−β 1 T)^2
```
```
(1−β 1 )^2
√
T(1−β 2 )
```
### ∑T

```
k=
```
### T

### (

```
β 12
√
β 2
```
```
)T−k
‖gk,i‖ 2
```
### ≤

### T∑− 1

```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### +

### T

### √

```
T(1−β 2 )
```
### ∑T

```
k=
```
```
γT−k‖gk,i‖ 2
```
Similarly, we can upper bound the rest of the terms in the summation.

```
∑T
```
```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### ≤

### ∑T

```
t=
```
```
‖gt,i‖ 2
√
t(1−β 2 )
```
```
T∑−t
```
```
j=
```
```
tγj
```
### ≤

### ∑T

```
t=
```
```
‖gt,i‖ 2
√
t(1−β 2 )
```
### ∑T

```
j=
```
```
tγj
```
Forγ < 1 , using the upper bound on the arithmetic-geometric series,

### ∑

```
ttγ
```
```
t< 1
(1−γ)^2 :
```
```
∑T
```
```
t=
```
```
‖gt,i‖ 2
√
t(1−β 2 )
```
### ∑T

```
j=
```
```
tγj≤
```
### 1

```
(1−γ)^2
```
### √

```
1 −β 2
```
### ∑T

```
t=
```
```
‖gt,i‖ 2
√
t
```
Apply Lemma 10.3,

```
∑T
```
```
t=
```
```
m̂^2 t,i
√
t̂vt,i
```
### ≤

### 2 G∞

```
(1−γ)^2
```
### √

```
1 −β 2
```
```
‖g1:T,i‖ 2
```
To simplify the notation, we defineγ, β
12
√β
2. Intuitively, our following theorem holds when the
learning rateαtis decaying at a rate oft−

(^12)
and first moment running average coefficientβ 1 ,tdecay
exponentially withλ, that is typically close to 1, e.g. 1 − 10 −^8.
Theorem 10.5.Assume that the functionfthas bounded gradients,‖∇ft(θ)‖ 2 ≤G,‖∇ft(θ)‖∞≤
G∞for allθ∈Rdand distance between anyθtgenerated by Adam is bounded,‖θn−θm‖ 2 ≤D,


‖θm−θn‖∞≤D∞for anym,n∈ { 1 ,...,T}, andβ 1 ,β 2 ∈[0,1)satisfy β

(^21)
√β
2 <^1. Letαt=
√α
t
andβ 1 ,t=β 1 λt−^1 ,λ∈(0,1). Adam achieves the following guarantee, for allT≥ 1.

### R(T)≤

### D^2

```
2 α(1−β 1 )
```
```
∑d
```
```
i=
```
### √

```
T̂vT,i+
```
```
α(β 1 + 1)G∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2 +
```
```
∑d
```
```
i=
```
### D^2 ∞G∞

### √

```
1 −β 2
2 α(1−β 1 )(1−λ)^2
```
Proof.Using Lemma 10.2, we have,

```
ft(θt)−ft(θ∗)≤gTt(θt−θ∗) =
```
```
∑d
```
```
i=
```
```
gt,i(θt,i−θ∗,i)
```
From the update rules presented in algorithm 1,

```
θt+1=θt−αtm̂t/
```
### √

```
v̂t
```
```
=θt−
```
```
αt
1 −βt 1
```
### (

```
β 1 ,t
√
̂vt
```
```
mt− 1 +
```
```
(1−β 1 ,t)
√
v̂t
```
```
gt
```
### )

We focus on theithdimension of the parameter vectorθt∈Rd. Subtract the scalarθ∗,iand square
both sides of the above update rule, we have,

(θt+1,i−θ∗,i)^2 =(θt,i−θ∗,i)^2 −

```
2 αt
1 −βt 1
```
### (

```
β 1 ,t
√
̂vt,i
```
```
mt− 1 ,i+ (1−
```
```
β 1 ,t)
√
̂vt,i
```
```
gt,i)(θt,i−θ∗,i) +α^2 t(
```
```
m̂t,i
√
̂vt,i
```
### )^2

We can rearrange the above equation and use Young’s inequality,ab≤a^2 /2 +b^2 / 2. Also, it can be

shown that

### √

```
̂vt,i=
```
### √∑

```
t
j=1(1−β^2 )β
```
```
t−j
2 g
```
```
2
j,i/
```
### √

```
1 −β 2 t≤‖g1:t,i‖ 2 andβ 1 ,t≤β 1. Then
```
gt,i(θt,i−θ∗,i) =

```
(1−βt 1 )
```
### √

```
̂vt,i
2 αt(1−β 1 ,t)
```
### (

```
(θt,i−θ,t∗)^2 −(θt+1,i−θ∗,i)^2
```
### )

### +

```
β 1 ,t
(1−β 1 ,t)
```
```
̂v
```
(^14)
t− 1 ,i
√
αt− 1
(θ∗,i−θt,i)

### √

```
αt− 1
```
```
mt− 1 ,i
```
```
̂v
```
(^14)
t− 1 ,i

### +

```
αt(1−β 1 t)
```
### √

```
̂vt,i
2(1−β 1 ,t)
```
### (

```
m̂t,i
√
̂vt,i
```
### )^2

### ≤

### 1

```
2 αt(1−β 1 )
```
### (

```
(θt,i−θ∗,t)^2 −(θt+1,i−θ∗,i)^2
```
### )√

```
̂vt,i+
```
```
β 1 ,t
2 αt− 1 (1−β 1 ,t)
```
```
(θ,i∗−θt,i)^2
```
### √

```
̂vt− 1 ,i
```
### +

```
β 1 αt− 1
2(1−β 1 )
```
```
m^2 t− 1 ,i
√
̂vt− 1 ,i
```
### +

```
αt
2(1−β 1 )
```
```
m̂^2 t,i
√
̂vt,i
```
We apply Lemma 10.4 to the above inequality and derive the regret bound by summing across all
the dimensions fori∈ 1 ,...,din the upper bound offt(θt)−ft(θ∗)and the sequence of convex
functions fort∈ 1 ,...,T:

### R(T)≤

```
∑d
```
```
i=
```
### 1

```
2 α 1 (1−β 1 )
```
```
(θ 1 ,i−θ∗,i)^2
```
### √

```
̂v 1 ,i+
```
```
∑d
```
```
i=
```
### ∑T

```
t=
```
### 1

```
2(1−β 1 )
```
```
(θt,i−θ∗,i)^2 (
```
### √

```
v̂t,i
αt
```
### −

### √

```
v̂t− 1 ,i
αt− 1
```
### )

### +

```
β 1 αG∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2 +
```
```
αG∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2
```
### +

```
∑d
```
```
i=
```
### ∑T

```
t=
```
```
β 1 ,t
2 αt(1−β 1 ,t)
```
```
(θ∗,i−θt,i)^2
```
### √

```
̂vt,i
```

From the assumption,‖θt−θ∗‖ 2 ≤D,‖θm−θn‖∞≤D∞, we have:

### R(T)≤

### D^2

```
2 α(1−β 1 )
```
```
∑d
```
```
i=
```
### √

```
T̂vT,i+
```
```
α(1 +β 1 )G∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2 +
```
### D^2 ∞

```
2 α
```
```
∑d
```
```
i=
```
```
∑t
```
```
t=
```
```
β 1 ,t
(1−β 1 ,t)
```
### √

```
t̂vt,i
```
### ≤

### D^2

```
2 α(1−β 1 )
```
```
∑d
```
```
i=
```
### √

```
T̂vT,i+
```
```
α(1 +β 1 )G∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2
```
### +

### D^2 ∞G∞

### √

```
1 −β 2
2 α
```
```
∑d
```
```
i=
```
```
∑t
```
```
t=
```
```
β 1 ,t
(1−β 1 ,t)
```
### √

```
t
```
We can use arithmetic geometric series upper bound for the last term:

```
∑t
```
```
t=
```
```
β 1 ,t
(1−β 1 ,t)
```
### √

```
t≤
```
```
∑t
```
```
t=
```
### 1

```
(1−β 1 )
```
```
λt−^1
```
### √

```
t
```
### ≤

```
∑t
```
```
t=
```
### 1

```
(1−β 1 )
```
```
λt−^1 t
```
### ≤

### 1

```
(1−β 1 )(1−λ)^2
```
Therefore, we have the following regret bound:

### R(T)≤

### D^2

```
2 α(1−β 1 )
```
```
∑d
```
```
i=
```
### √

```
T̂vT,i+
```
```
α(1 +β 1 )G∞
(1−β 1 )
```
### √

```
1 −β 2 (1−γ)^2
```
```
∑d
```
```
i=
```
```
‖g1:T,i‖ 2 +
```
```
∑d
```
```
i=
```
### D∞^2 G∞

### √

```
1 −β 2
2 αβ 1 (1−λ)^2
```

