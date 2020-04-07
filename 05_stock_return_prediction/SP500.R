## Mid-term project: Predicting stock market returns
## Group 3 coder: Xiaoyi Chen
## Reference: Data Mining with R: Learning with Case Studies
## Description: We are aimed to forecast the price of the S&P 500 index according to the historical
##              quotes data and to provide trade decisions based on the preditions of the model. 


# Loading data
# Here we load data from the package. Alternatively, you can load from csv file,
# from MySQL Database or from the web.

library(DMwR)# This is a package from the reference book
data(GSPC)

# Overviewing the dataset.
# It's a time series data describing the price and volume of S&P 500 index from 1970 to 2009.
head(GSPC)
class(GSPC)
summary(GSPC)

########################################################################################
####                              Problem Formulation                               ####
########################################################################################

# Defining the calculation function of target variable T indicator
# Input: quotes - historical time series
#        tgt.margin - targeted variable margin
#        n.days - time window for predition, i.e. we are predicting for the next n.days days
# Output: T.ind - an indicator for buying or selling 
# (Highly positive value suggests buy orders and highly negative value suggests sell orders)

T.ind <- function(quotes, tgt.margin = 0.025, n.days = 10) {
  
  # Calculating the daily average price as the average of high, low and closed prices
  v <- apply(HLC(quotes), 1, mean)
  
  # Calculating the return percentage from current time to each day in the next n.days days
  r <- matrix(NA,ncol=n.days,nrow=NROW(quotes))
  ## This statement in the book is wrong, so I change it
  for(x in 1:n.days) 
    r[,x] <- Next(Delt(Cl(quotes),v,k=x),x)
  
  # Summing up all returns that are larger than tgt.margin
  x <- apply(r, 1, function(x) sum(x[x > tgt.margin | x < -tgt.margin]))
  
  # Transforming the results to a time series
  if (is.xts(quotes))
    xts(x, time(quotes))
  else x
}

# Visualizing

library(quantmod)
# Drawing a candlestick graph for the quotes of last 3 months
candleChart(last(GSPC, "3 months"), theme = "white", TA = NULL)
# Calculating the average price of each day
avgPrice <- function(p) apply(HLC(p), 1, mean)
# Adding average prices and T indicators to the candlestick graph
addAvgPrice <- newTA(FUN = avgPrice, col = 1, legend = "AvgPrice")
addT.ind <- newTA(FUN = T.ind, col = "red", legend = "tgtRet")
addAvgPrice(on = 1) # parameter "on" defines where to draw in the graph
# The T indicator summarizes the variations in the following ten days.
addT.ind()

# Selecting technical indicators, it is a problem of feature selection

# Defining the initial feature sets
# All of the following indicators can be implemented through functions in package TTR
library(TTR)
# Average True Range (ATR)
myATR <- function(x) ATR(HLC(x))[, "atr"]
# Stochastic Momentum Index (SMI)
mySMI <- function(x) SMI(HLC(x))[, "SMI"]
# Welles Wilder’s Directional Movement Index (ADX)
myADX <- function(x) ADX(HLC(x))[, "ADX"]
# Aroon indicator
myAroon <- function(x) aroon(x[, c("High", "Low")])$oscillator
# Bollinger Bands
myBB <- function(x) BBands(HLC(x))[, "pctB"]
# Chaikin Volatility
myChaikinVol <- function(x) Delt(chaikinVolatility(x[, c("High","Low")]))[, 1]
# Close Location Value (CLV) 
myCLV <- function(x) EMA(CLV(HLC(x)))[, 1]
# Arms’ Ease of Movement Value (EMV)
myEMV <- function(x) EMV(x[, c("High", "Low")], x[, "Volume"])[,2]
# MACD oscillator
myMACD <- function(x) MACD(Cl(x))[, 2]
# Money Flow Index (MFI)
myMFI <- function(x) MFI(x[, c("High", "Low", "Close")],x[, "Volume"])
# Parabolic Stop-and-Reverse
mySAR <- function(x) SAR(x[, c("High", "Close")])[, 1]
# Volatility indicator
myVolat <- function(x) volatility(OHLC(x), calc = "garman")[,1]

# Selecting features through random forest
# Resetting the dataset
data(GSPC)
library(randomForest)

# Specifying data with initial feature sets
data.model <- specifyModel(T.ind(GSPC) ~ Delt(Cl(GSPC),k=1:10) +
                             myATR(GSPC) + mySMI(GSPC) + myADX(GSPC) + myAroon(GSPC) +
                             myBB(GSPC)  + myChaikinVol(GSPC) + myCLV(GSPC) +
                             CMO(Cl(GSPC)) + EMA(Delt(Cl(GSPC))) + myEMV(GSPC) +
                             myVolat(GSPC)  + myMACD(GSPC) + myMFI(GSPC) + RSI(Cl(GSPC)) +
                             mySAR(GSPC) + runMean(Cl(GSPC)) + runSD(Cl(GSPC)))
# Showing the model formula
data.model@model.formula

# Building the random forest model through a training set with the data of first 30 years
set.seed(1234)
rf <- buildModel(data.model,method='randomForest', 
                 training.per=c(start(GSPC),index(GSPC["1999-12-31"])), 
                 ntree=50, importance=T)

# A simple example to show the application of above functions
# Specifying and obtaining data
ex.model <- specifyModel(T.ind(IBM) ~ Delt(Cl(IBM), k = 1:3))
data <- modelData(ex.model, data.window = c("2009-01-01", "2009-08-10"))
# Then you can apply the model formula and your data to arbitary models
ex.model@model.formula

# Evaluating the importance of each feature
# Here the importance of feature is defined as the average increasement of error measure
# when each feature is removed.
# So the higher the importance, the more important of the feature.

# Plotting the feature importance in an decreasing order
varImpPlot(rf@fitted.model, type = 1,cex=0.7)

# Setting the importance threshold to 10
# So we will eliminate features with importance less than 10
imp <- importance(rf@fitted.model, type = 1)
rownames(imp)[which(imp > 10)]

# Transforming xts objects to data frame (because we need add categorical variables for classification)
data.model <- specifyModel(T.ind(GSPC) ~ Delt(Cl(GSPC),k=1) + myATR(GSPC) + myADX(GSPC) +    myEMV(GSPC) + myVolat(GSPC)  + myMACD(GSPC) + mySAR(GSPC) + runMean(Cl(GSPC)) )

# Training set: data from the first 30 years
Tdata.train <- as.data.frame(modelData(data.model, data.window=c('1970-01-02','1999-12-31'))) 
# Testing set: data from the last 10 years
Tdata.eval <- na.omit(as.data.frame(modelData(data.model,data.window=c('2000-01-01','2009-09-15')))) 


########################################################################################
####                                     Modeling                                   ####
########################################################################################


# Firstly, we consider a regression problem, i.e., predicting the value of T indicator
# and then transforming the value into trading signal.

# Defining the prediction function
# Setting T indicator as the target and the other varaibles as predictors
Tform <- as.formula('T.ind.GSPC ~ .')

# Applying ANN to regression problem

# Setting a random seed to make sure that the initial weights of ANN keep constant
set.seed(1234)
library(nnet)
# Normalizing the data to make all variables have a mean of zero and a deviation of one
norm.data <- scale(Tdata.train)

# Using the first 1000 training data bo build a network with 10 nodes in the hidden layer
nn <- nnet(Tform, norm.data[1:1000, ], size = 10, decay = 0.01,
           maxit = 1000, linout = T, trace = F)
# testing the model on the following 1000 data
norm.preds <- predict(nn, norm.data[1001:2000, ])
# Unscaling the prediction values with the mean and deviation used above scaling process
preds <- unscale(norm.preds, norm.data)

# Evaluating the ANN prediction

# Transforming the both predition and true values of T indicator into trading signals
# with the thresholds of -0.1 and 0.1. That is, if T_ind < -0.1, sell; 
# if T_ind > 0.1, buy; others, hold.
sigs.nn <- trading.signals(preds, 0.1, -0.1)
true.sigs <- trading.signals(Tdata.train[1001:2000, "T.ind.GSPC"], 0.1, -0.1)

# Calculating the precision and recall scores of each event
# where s refers to sell, b refers to buy and s+b refers to hold
# Precision is proportion of event signals produced by the event actually happens.
# Recall is the proportion of events occurring that is signaled as such.
# For both scores, the higher, the better.
# The result is relatively low. The low precision score mean that the model gives 
# wrong signals and the low recall scores mean that the model lose some opportunities.
sigs.PR(sigs.nn, true.sigs)

# Now, we consider the classification problem, i.e., transforming the T indicator 
# in training dataset into categorical trade signal and predicting the signal directly.

# Transforming T indicators into trade signals with threshhold of -0.1 and 0.1
signals <- trading.signals(Tdata.train[, "T.ind.GSPC"], 0.1,-0.1)
# Normalizing other continuous variables
norm.data <- data.frame(signals = signals, scale(Tdata.train[,-1]))

# Applying ANN to the classification problem
set.seed(1234)
library(nnet)
nn <- nnet(signals ~ ., norm.data[1:1000, ], size = 10, decay = 0.01,
           maxit = 1000, trace = F)
preds <- predict(nn, norm.data[1001:2000, ], type = "class")
# The precision scores are higher than that in the regression problem, however the recall
# scores are lower and generally both of them are not high enough.
sigs.PR(preds, norm.data[1001:2000, 1])

# Applying SVM to regression problem
library(e1071)
# Building a SVM model on the first 1000 training data
sv <- svm(Tform, Tdata.train[1:1000, ], gamma = 0.01, cost = 10,class.weights=c())
# Predicting the T indicators for the following 1000 training data
s.preds <- predict(sv, Tdata.train[1001:2000, ])
# Transforming T indicators to trade signals
sigs.svm <- trading.signals(s.preds, 0.1, -0.1)
true.sigs <- trading.signals(Tdata.train[1001:2000, "T.ind.GSPC"],0.1, -0.1)
# Calculating the precision and recall scores
# The precision scores of SVM are higher than ANN,however, the recall scores are less.
# Note that the SVM may suffer from imbanlanced datasets, which leads to high precisions yet low recalls.
sigs.PR(sigs.svm, true.sigs)

# Applying SVM to the classification problem
library(kernlab)
data <- cbind(signals = signals, Tdata.train[, -1])
ksv <- ksvm(signals ~ ., data[1:1000, ], C = 10)
ks.preds <- predict(ksv, data[1001:2000, ])
# The precision result is not as good as the regression problem but 
# the recall score is better. 
sigs.PR(ks.preds, data[1001:2000, 1])

# Applying multivariate adaptive regression model to the regression problem
# Note that the model is only available for regression problems
library(earth)
e <- earth(Tform, Tdata.train[1:1000, ])
e.preds <- predict(e, Tdata.train[1001:2000, ])
sigs.e <- trading.signals(e.preds, 0.1, -0.1)
true.sigs <- trading.signals(Tdata.train[1001:2000, "T.ind.GSPC"], 0.1, -0.1)
sigs.PR(sigs.e, true.sigs)


########################################################################################
####                                Trade Policy                                    ####
########################################################################################

# Defining a function to simulate the first trade policy
# In the first policy, we consider open a long position with a buy signal and 
# open a short position with a sell signal. However, only one position is opened at any time
# When we open a new position, there are three orders: a market order to open the position,
# a limit order with a target price and a stop order to limit loss.

# Input: signals - predicted signals until the current day
#        market - market quotes until the current day
#        opened.pos - the currently opened positions in a four-column matrix
#        money - the money currently available
#        bet - percentage of current money that invest in a new position
#        exp.prof - profit margin
#        hold.time - the number of days that we wait to reach the profit margin
#        max.loss - the maximum loss before closing the position
# Output: policy.1 - a five-column including order direction, order type,
#                    trade quantity, action and position ID.

policy.1 <- function(signals,market,opened.pos,money,
                     bet=0.2,hold.time=10,
                     exp.prof=0.025, max.loss= 0.05)
{
  d <- NROW(market) # getting the ID of the current day
  orders <- NULL
  nOs <- NROW(opened.pos) # getting the postions have already opened
  
  # i) hold signal: nothing to do
  if (!nOs && signals[d] == 'h') return(orders)
  
  # ii) buy signal: long positions 
  if (signals[d] == 'b' && !nOs) {
    # Calculating the availabe quantity of stocks to trade 
    quant <- round(bet*money/market[d,'Close'],0) 
    if (quant > 0)
      # Add a new row in orders
      orders <- rbind(orders,
                      data.frame(order=c(1,-1,-1), #1 for buy orders and -1 for sell orders
                                 order.type=c(1,2,3), 
                                 #1 for immediate orders,2 for limit orders and 3 for stop orders  
                                 val = c(quant,
                                         market[d,'Close']*(1+exp.prof),# target price in sell order
                                         market[d,'Close']*(1-max.loss)# the limit in sell stop order
                                         ),
                                 action = c('open','close','close'),
                                 posID=c(NA,NA,NA)
                      )
      )
  }
  
  # ii) sell signal: short positions
  else if (signals[d] == 's' && !nOs) {
    # Calculating the quantity of stocks we already need to buy
    # because of currently opened short positions
    need2buy <- sum(opened.pos[opened.pos[,'pos.type']==-1,
                               "N.stocks"])*market[d,'Close']
    # Calculating the remaining quantity of stocks to trade
    quant<-round(bet*(money-need2buy)/market[d,'Close'],0)
    if(quant>0)
      orders<-rbind(orders,
                    data.frame(order=c(-1,1,1),order.type=c(1,2,3),
                               val = c(quant, 
                                       market[d,'Close']*(1-exp.prof), #target price in buy order
                                       market[d,'Close']*(1+max.loss) # the limit in buy stop order
                               ),
                               action = c('open','close','close'),
                               posID = c(NA,NA,NA)
                               )
                    )
  }
  
# If we already have positions opened, check if we need to close them
if (nOs)
  for(i in 1:nOs) {
    if (d - opened.pos[i,'Odate'] >= hold.time) #if the holding time is over 
      orders <- rbind(orders, 
                      data.frame(order=-opened.pos[i,'pos.type'],
                                 # change to the opposite order action
                                 order.type=1,
                                 # carry out immediately
                                 val = NA,
                                 action = 'close',
                                 # close the position 
                                 posID = rownames(opened.pos)[i]
                      )
      )
  }
  orders
}

# Defining a function to simulate the second trade policy
# In the second policy, we have similiar strategies with the first one, 
# but we can always open new positions if we have relevant signals and sufficient money.
# In addition, we don't have a hold time, which means we will wait forever for a target price.

# The function is generally the same as the above one except for removing the limitation
# of opened position and removing the close operation.

policy.2 <- function(signals,market,opened.pos,money,
                     bet=0.2,exp.prof=0.025, max.loss= 0.05)
{
  d <- NROW(market) # this is the ID of today
  orders <- NULL
  nOs <- NROW(opened.pos)
  # nothing to do
  if (!nOs && signals[d] == 'h') return(orders)
  
  # First lets check if we can open new positions 
  # i) long positions
  if (signals[d] == 'b') {
    quant <- round(bet*money/market[d,'Close'],0) 
    if (quant > 0)
      orders <- rbind(orders,
                      data.frame(order=c(1,-1,-1),order.type=c(1,2,3),
                                 val = c(quant,
                                         market[d,'Close']*(1+exp.prof),
                                         market[d,'Close']*(1-max.loss)
                                 ),
                                 action = c('open','close','close'),
                                 posID = c(NA,NA,NA)
                      )
      )
    }
  # ii) short positions
  else if (signals[d] == 's') {
    # this is the money already committed to buy stocks
    # because of currently opened short positions
    need2buy <- sum(opened.pos[opened.pos[,'pos.type']==-1,
                               "N.stocks"])*market[d,'Close'] 
    quant <- round(bet*(money-need2buy)/market[d,'Close'],0)
    if (quant > 0)
      orders <- rbind(orders,
                      data.frame(order=c(-1,1,1),order.type=c(1,2,3),
                                 val = c(quant,
                                         market[d,'Close']*(1-exp.prof), 
                                         market[d,'Close']*(1+max.loss)
                                         ),
                                 action = c('open','close','close'),
                                 posID = c(NA,NA,NA)
                                 )
      )
    }

  orders
}


# Using ANN to train a prediction model and simulating defined policy functions

# For illustraing, setting lengh of training set to 1000 and testing set to 500
start <- 1
len.tr <- 1000
len.ts <- 500
tr <- start:(start+len.tr-1)
ts <- (start+len.tr):(start+len.tr+len.ts-1)

# Obtaining the market quotes for the testing period
data(GSPC)
date <- rownames(Tdata.train[start+len.tr,])
market <- GSPC[paste(date,'/',sep='')][1:len.ts]

## Learning SVM model and obtaining signal predictions
#library(e1071)
#s <- svm(Tform,Tdata.train[tr,],cost=10,gamma=0.01)
#p <- predict(s,Tdata.train[ts,])
#sig <- trading.signals(p,0.1,-0.1)

# Learning ANN model and obtaining signal predictions
library(nnet)
set.seed(1234)
s <- nnet(Tform,norm.data[tr,],size=10,decay=0.01,
          maxit=1000,linout=T,trace=F)
norm.p <- predict(s,norm.data[ts,])
p<-unscale(norm.p,norm.data)
sig <- trading.signals(p,0.1,-0.1)


# Simulating the first policy trader
t1 <- trading.simulator(market,sig,
                        'policy.1',list(exp.prof=0.05,bet=0.2,hold.time=30))
# Printing trade record
t1
# Viewing the trade summary. The return is 0.91%, it's not good.
summary(t1)
# Viewing economic performance indicators
tradingEvaluation(t1)
# Visualizing the performance
plot(t1, market, theme = "white", name = "SP500, t1(ANN, Policy 1)",cex=0.5)

# Simulating the second policy trader
t2 <- trading.simulator(market, sig, "policy.2", list(exp.prof = 0.05,bet = 0.3))
# The return is -2.4%.
summary(t2)
plot(t2, market, theme = "white", name = "SP500, t2")

# Repeating the experiment with a different training and testing dataset
start <- 2000
len.tr <- 1000
len.ts <- 500
tr <- start:(start + len.tr - 1)
ts <- (start + len.tr):(start + len.tr + len.ts - 1)

#s <- svm(Tform, Tdata.train[tr, ], cost = 10, gamma = 0.01)
#p <- predict(s, Tdata.train[ts, ])
set.seed(1234)
s <- nnet(Tform,norm.data[tr,],size=10,decay=0.01,
          maxit=1000,linout=T,trace=F)
norm.p <- predict(s,norm.data[ts,])
p<-unscale(norm.p,norm.data)

sig <- trading.signals(p, 0.1, -0.1)
t3 <- trading.simulator(market, sig, "policy.2", list(exp.prof = 0.05,bet = 0.3))
# The return is -90.69%
# We can learn from the different results that we need more reliable estimate.
# So next we will run Monte Carlo estimation.
summary(t3)

summary(t1)
summary(t2)
summary(t3)
tradingEvaluation(t1)
tradingEvaluation(t2)
tradingEvaluation(t3)


########################################################################################
####                                Monte Carlo Simulation                          ####
########################################################################################


# Defining some required functions in sumulation
# form: model formula
# train: training set, test: testing set
# b.t: buy threshold, s.t: sell threshold

# SVM for regression
MC.svmR <- function(form, train, test, b.t = 0.1, s.t = -0.1,...) {
  require(e1071)
  t <- svm(form, train, ...) #training
  p <- predict(t, test)  #predicting
  trading.signals(p, b.t, s.t) #discretizing into signals
}

# SVM for classification
MC.svmC <- function(form, train, test, b.t = 0.1, s.t = -0.1,...) {
  require(e1071)
  tgtName <- all.vars(form)[1]
  train[, tgtName] <- trading.signals(train[, tgtName],b.t, s.t) # Obtaining train signals
  t <- svm(form, train, ...) #training
  p <- predict(t, test) #predicting
  factor(p, levels = c("s", "h", "b")) #labeling prediction
}

# ANN for regression
MC.nnetR <- function(form, train, test, b.t = 0.1, s.t = -0.1,...) {
  require(nnet)
  t <- nnet(form, train, ...) #training
  p <- predict(t, test) #predicting
  trading.signals(p, b.t, s.t) #discretizing
}

# ANN for classification
MC.nnetC <- function(form, train, test, b.t = 0.1, s.t = -0.1,...) {
  require(nnet)
  tgtName <- all.vars(form)[1]
  train[, tgtName] <- trading.signals(train[, tgtName],b.t, s.t) # obtaining train signals
  t <- nnet(form, train, ...) # training
  p <- predict(t, test, type = "class") #predicting
  factor(p, levels = c("s", "h", "b")) #labeling
}

# Multivariate adaptive regression model
MC.earth <- function(form, train, test, b.t = 0.1, s.t = -0.1,...) {
  require(earth)
  t <- earth(form, train, ...) #training
  p <- predict(t, test) #predicting
  trading.signals(p, b.t, s.t) #discretizing
}

# Performing a single simulated trader
# learner: learning model, policy.func: specified trading strategy
#single <- function(form, train, test, learner, policy.func, ...) {
#  p <- do.call(paste("MC", learner, sep = "."), list(form,train, test, ...)) # learning the model
#  eval.stats(form, train, test, p, policy.func = policy.func) # trading and evaluating
#}
singleModel <- function(form,train,test,learner,policy.func,...) {
  p <- do.call(paste('MC',learner,sep='.'),list(form,train,test,...))
  eval.stats(form,train,test,p,policy.func=policy.func)
}

# Sliding window simulation: update new data and delete old data 
# so that the training size is constant.
# relearn.step: window size to update training data
slide <- function(form, train, test, learner, relearn.step,policy.func, ...){
  real.learner <- learner(paste("MC", learner, sep = "."), 
                          pars = list(...)) # specifying a learner
  p <- slidingWindowTest(real.learner, form, train, test, 
                         relearn.step) # learning the model by sliding window method
  p <- factor(p, levels = 1:3, labels = c("s", "h", "b")) # labeling
  eval.stats(form, train, test, p, policy.func = policy.func) # trading and evaluating
}

# Growing window simulation: update new data without deleting old data 
# so that the training size is increasing.
grow <- function(form, train, test, learner, relearn.step,policy.func, ...) {
 real.learner <- learner(paste("MC", learner, sep = "."),
                          pars = list(...)) # specifying a learner
  p <- growingWindowTest(real.learner, form, train, test,
                         relearn.step) # learning the model by growing window method
  p <- factor(p, levels = 1:3, labels = c("s", "h", "b")) # labeling
  eval.stats(form, train, test, p, policy.func = policy.func) # trading and evaluating
}

# Performing a simulated trader and evaluating performance 
# returning precision and recall of signals and economic metrics of trading
# preds: predicted signals
eval.stats <- function(form,train,test,preds,b.t=0.1,s.t=-0.1,...) {
  # Signals evaluation
  tgtName <- all.vars(form)[1]
  test[,tgtName] <- trading.signals(test[,tgtName],b.t,s.t) # Obtaining true signals in testing set
  st <- sigs.PR(preds,test[,tgtName]) # Comparing prediction signals with true signals
  dim(st) <- NULL
  names(st) <- paste(rep(c('prec','rec'),each=3), # Obtaining precision and recall
                     c('s','b','sb'),sep='.')
  # Trading evaluation
  date <- rownames(test)[1]
  market <- GSPC[paste(date,"/",sep='')][1:length(preds),] # Obtaining market quotes
  trade.res <- trading.simulator(market,preds,...) # performing a simulated trader
  
  c(st,tradingEvaluation(trade.res)) # Obtaining trade performance
}

# Generating three variants from trading policies described above
# These variants are assigned to different parameters

pol1 <- function(signals,market,op,money)
  policy.1(signals,market,op,money,
           bet=0.2,exp.prof=0.025,max.loss=0.05,hold.time=10)

pol2 <- function(signals,market,op,money)
  policy.1(signals,market,op,money,
           bet=0.2,exp.prof=0.05,max.loss=0.05,hold.time=20)

pol3 <- function(signals,market,op,money)
  policy.2(signals,market,op,money,
         bet=0.5,exp.prof=0.05,max.loss=0.05)

######### Running Monte Carlo experiments ########

# The list of learners we will use
TODO <- c('svmR','svmC','earth','nnetR','nnetC')
# The datasets used in the comparison
DSs <- list(dataset(Tform,Tdata.train,'SP500')) 
# Monte Carlo (MC) settings
MCsetts <- mcSettings(20, # 20 repetitions of the MC exps
                      2540,# ~ 10 years for training
                      1270, # ~ 5 years for testing
                      1234 # random number generator seed
                      )
# Variants to try for all learners
VARS <- list()
VARS$svmR<-list(cost=c(10,150),gamma=c(0.01,0.001), # gamma is a parameter for svm kernels
                policy.func=c('pol1','pol2','pol3'))
VARS$svmC<-list(cost=c(10,150),gamma=c(0.01,0.001), 
                policy.func=c('pol1','pol2','pol3'))
VARS$earth <- list(nk=c(10,17),degree=c(1,2),thresh=c(0.01,0.001), 
                   policy.func=c('pol1','pol2','pol3'))
VARS$nnetR <- list(linout=T,maxit=750,size=c(5,10),
                   decay=c(0.001,0.01),
                   policy.func=c('pol1','pol2','pol3'))
VARS$nnetC <- list(maxit=750,size=c(5,10),decay=c(0.001,0.01),
                   policy.func=c('pol1','pol2','pol3')) 

# main loop

#for(td in TODO) {
#  assign(td, # learner
#         experimentalComparison(
#           DSs, # dataset
#           c(
#             do.call('variants', 
#              c(list('singleModel',learner=td),VARS[[td]],
#                            varsRootName=paste('singleModel',td,sep='.'))), 
#             do.call('variants',
#              c(list('slide',learner=td, 
#                     relearn.step=c(60,120)),
#                VARS[[td]],
#                varsRootName=paste('slide',td,sep='.'))),
#             do.call('variants',
#              c(list('grow',learner=td, relearn.step=c(60,120)),
#                VARS[[td]], varsRootName=paste('grow',td,sep='.')))
#             ),
#          MCsetts)
#         )
#  # save the results
#  save(list=td,file=paste(td,'Rdata',sep='.'))
#}


########################################################################################
####                                    Evaluation                                  ####
########################################################################################

# Loading results
load("/Users/monica/Documents/[Rutgers]Study/2018fall/AnalyticsBusIntell/GroupAssignment/MidTerm/svmR.Rdata")
load("/Users/monica/Documents/[Rutgers]Study/2018fall/AnalyticsBusIntell/GroupAssignment/MidTerm/svmC.Rdata")
load("/Users/monica/Documents/[Rutgers]Study/2018fall/AnalyticsBusIntell/GroupAssignment/MidTerm/earth.Rdata")
load("/Users/monica/Documents/[Rutgers]Study/2018fall/AnalyticsBusIntell/GroupAssignment/MidTerm/nnetR.Rdata")
load("/Users/monica/Documents/[Rutgers]Study/2018fall/AnalyticsBusIntell/GroupAssignment/MidTerm/nnetC.Rdata")

# Identifying the targeted statistics

# Signal prediction performance: prefer precision, larger preferred
#    precision: wrong signals on opening a new position--> increase cost
#    recall: fail to capture trading opportunity --> lose opportunity
# Trading performance: 
#    Ret: return, larger preferred
#    PercProf: percentage of profitable trades (should >50%), larger preferred
#    MaxDD: maximum draw-down, lower preferred
#    SharpeRatio: sharp ratio, larger preferred
tgtStats <- c('prec.sb','Ret','PercProf','MaxDD','SharpeRatio')

# Selecting related information from the simulated results
allSysRes <- join(subset(svmR,stats=tgtStats),
                  subset(svmC,stats=tgtStats),
                  subset(nnetR,stats=tgtStats),
                  subset(nnetC,stats=tgtStats),
                  subset(earth,stats=tgtStats),
                  by = 'variants')

# Showing top 5 trading systems in terms of each statistic.
# From the rank, we can find: 1) all of them involve either svm or ann
# 2) most of them use window mechanism (rather than single)
# 3) However, some scores are too ideal and therefore suspicious.
# For example, the precision of SVM are pretty high and maxDD of SVM are zero
# This results from imbalance dataset. The result makes no sense for trading.
# 4) The sharp ratio is too low. What does it mean?
rankSystems(allSysRes,5,maxs=c(T,T,T,F,T))

# Checking the first two systems in the rank of precision
# The average return are 0.15% and 0.23%, -76% below the naive trategy,
# which means these models are useless.
summary(subset(svmR,
               stats=c('Ret','RetOverBH','PercProf','NTrades'),
               vars=c('grow.svmR.v5','grow.svmR.v13')))

# Adding some constraints on statistics
fullResults <- join(svmR, svmC, earth, nnetC, nnetR, by = "variants")
nt <- statScores(fullResults, "NTrades")[[1]] # number of average trade
rt <- statScores(fullResults, "Ret")[[1]] # average return
pp <- statScores(fullResults, "PercProf")[[1]] # percentage of profitable trades
s1 <- names(nt)[which(nt > 20)] #average trade >20
s2 <- names(rt)[which(rt > 0.5)] # average return>0.5%
s3 <- names(pp)[which(pp > 40)] # % of profitable trades >40%
namesBest <- intersect(intersect(s1, s2), s3)
# The trading systems that satisfy all of the three constraints
# All of them use the regression algorithm of ANN
# However, the training data are used differently.
namesBest
# Showing details of three systems
# singleModel.nnetC.v11: the average return is remarkable (35.58%), 
#                   however the minimal is -100.59%
#                   and the std is 262%. The result is instable.
# slide.nnetR.v11 & grow.nnetR.v18: more stable yet lower return.
summary(subset(fullResults,
                 stats=tgtStats,
                 vars=namesBest))

# Analyzing statistical significance
compAnalysis(subset(fullResults,
                    stats=tgtStats,
                    vars=namesBest))

# Visualizing the distribution of selected scores across 20 repititions
# The scores of singleModel.nnetC.v11 are significantly different from the others.
# We can find an outlier in singleModel.nnetC.v11, which lead to high average return.
plot(subset(fullResults,
            stats=c('Ret','PercProf','MaxDD'), 
            vars=namesBest))

# Checking details in singleModel.nnetC.v11 system. 
# This system shouldn't be selected because it is instable.
getVariant("singleModel.nnetC.v11", nnetC)


########################################################################################
####                               Final Tading System                              ####
########################################################################################

# Validating the selected two systems on testing dataset (data in the last 9 years)
data <- tail(Tdata.train, 2540)
results <- list()
for (name in namesBest[3]) {
  set.seed(1234)
  sys <- getVariant(name, fullResults)
  if (sys@func=="single")
    sys@func="singleModel"
  results[[name]] <- runLearner(sys, Tform, data, Tdata.eval)
}
# Showing statistics of selected models
# The best model is slide.nnetR.v11
results <- t(as.data.frame(results))
results[, c("Ret", "RetOverBH", "MaxDD", "SharpeRatio", "NTrades",
            "PercProf")]

# Showing configuration of the best model
getVariant("slide.nnetR.v11", fullResults)

# Obtaining the trading record
# Specifying the model parameters
model <- learner("MC.nnetR", list(maxit = 750, linout = T,
                                  trace = F, size = 10, decay = 0.001))
# Predicting signals
set.seed(1234)
preds <- slidingWindowTest(model, Tform, data, Tdata.eval,
                           relearn.step = 60)
signals <- factor(preds, levels = 1:3, labels = c("s", "h", "b"))
# Obtaining market quotes
date <- rownames(Tdata.eval)[1]
market <- GSPC[paste(date, "/", sep = "")][1:length(signals),]
# Simulating the best trader
set.seed(1234)
trade.res <- trading.simulator(market, signals, policy.func = "pol2")

# Ploting the performance
plot(trade.res, market, theme = "white", name = "SP500 - final test")

# Evaluating the performance of final system
library(PerformanceAnalytics)
# Calculating returns
rets <- Return.calculate(trade.res@trading$Equity)
# Visualizing returns
chart.CumReturns(rets, main = "Cumulative returns of the strategy", ylab = "returns")

# Printing annual return
# Converting zoo matrix to one-dimension xts vector; otherwise, there will be an error
xxxxxx<-as.xts(trade.res@trading$Equity) 
yearlyReturn(xxxxxx)

# Visulazing annual return
# Only two years with negative returns
plot(100*yearlyReturn(xxxxxx),
     main='Yearly percentage returns of the trading system')
abline(h=0,lty=2)

# Showing more specific return values
dim(rets) <- c(NROW(rets),NCOL(rets))
colnames(rets)<-"rets"
index(rets) <- as.Date(index(rets), origin="2000-01-03")
#rets[1:50]
table.CalendarReturns(rets)

# Showing information concerning the risk analysis of the strategy 
table.DownsideRisk(rets)
