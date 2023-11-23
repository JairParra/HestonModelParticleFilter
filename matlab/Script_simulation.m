%Author Rémi Galarneau-Vincent

%% Simulated observations
clear all, clc, close all

%
% Calibrate the number of days
num_days = 1000; % Number of days
%

% Chose a set of parameters
mu=0.06; %drift
kappa = 5; % speed of mean reversion
theta = 0.02; % Mean reversion level
rho = -0.7; % Correlation
sigma = 0.3; % volatility of volatility
v=0.01;
%
% Simulation of returns
[ret, V, RV] = simulate_Heston93(mu, kappa, theta, sigma, rho, v, num_days);

% Transformed parameters
mu = mu*10;
sigma = log(sigma);
kappa = log(kappa);
theta = log(theta);
rho   = log(1+rho)-log(1-rho);
eta   = log(0.2);
x =[mu, kappa, theta, sigma , rho, eta];

% Generating the random variables
nb_part = 10000;
N = length(ret);
simul.U = unifrnd(0,1,nb_part,1);
simul.ZZ = normrnd(0,1,nb_part,num_days,5);
simul.UU =  unifrnd(0,1,nb_part,num_days);
simul.Z = normrnd(0,1,nb_part,num_days);

% Filter using only the returns and the true parameters
[Like_H_true_ret,V_ret_true]=Heston93(x,simul, ret,1);


%Filter using the returns, the Realized Variance and the true parameters
tic;
[Like_H_true_rv,V_RV_true]=Heston93(x,simul, [ret, RV],1);
toc;

mu=0.06; %drift
kappa = 5; % speed of mean reversion
theta = 0.02; % Mean reversion level
rho = -0.7; % Correlation
sigma = 0.3; % volatility of volatility
eta   = 0.2;

% Transformed parameters
mu = mu*10;
sigma = log(sigma);
kappa = log(kappa);
theta = log(theta);
rho   = log(1+rho)-log(1-rho);
eta   = log(eta);
x1 =[mu, kappa, theta, sigma , rho];
x2 =[mu, kappa, theta, sigma , rho, eta];


% Parameter search I use fminsearch for stability
options = optimset('Display','iter','PlotFcns',@optimplotfval,'TolFun',1e-3,'TolX',1e-3);
Opt_ret=fminsearch(@(x) Heston93(x,simul, ret),x1,options);
Opt_rv=fminsearch(@(x) Heston93(x,simul, [ret, RV]),x2,options);

% Compute the filtered vlatility and
[Like_H_ret,V_ret_est]=Heston93(Opt_ret,simul, ret,1);
[Like_H_rv,V_RV_est]=Heston93(Opt_rv,simul, [ret, RV],1);

%% Graph of returns volatility and volatility filtered/ Only returns
figure
tiledlayout(4,1) 
nexttile
plot(ret,'k')
xlabel('days')
ylabel('log-returns')
title('Simulated returns')
xlim([0,num_days])
nexttile
plot(V,'k')
hold on
plot(V_ret_true(:,1),'Color','b')
plot(V_ret_true(:,2),'Color' ,'r','LineStyle','--')
plot(V_ret_true(:,3),'Color' ,'r','LineStyle','--')
title('Volatility filtered using returns and the true parameters')
xlabel('days')
ylabel('volatility level')
legend('True variance','Filtered variance','2.5%','97.5%','Location','northwest')
xlim([0,num_days])
nexttile
plot(V,'k')
hold on
plot(V_ret_est(:,1),'Color','b')
plot(V_ret_est(:,2),'Color' ,'r','LineStyle','--')
plot(V_ret_est(:,3),'Color' ,'r','LineStyle','--')
legend('True variance','Filtered variance','2.5%','97.5%','Location','northwest')
title('Volatility filtered using returns and the estimated parameters')
xlim([0,num_days])
nexttile
plot(V,'k')
hold on
plot(V_ret_true(:,1),'Color' ,'r')
plot(V_ret_est(:,1),'Color','b')
legend('True variance','Filtered variance true parameters','Filtered Variance estimated parameters','Location','northwest')
xlim([0,num_days])
title('Volatility filtered using the true and estimated parameters')

%% Graph of returns volatility and volatility filtered/ Returns and RV
figure
tiledlayout(4,1) 
nexttile
plot(ret,'k')
xlabel('days')
ylabel('log-returns')
title('Simulated returns')
xlim([0,num_days])
nexttile
plot(V,'k')
hold on
plot(V_RV_true(:,1),'Color','b')
plot(V_RV_true(:,2),'Color' ,'r','LineStyle','--')
plot(V_RV_true(:,3),'Color' ,'r','LineStyle','--')
title('Volatility filtered using returns + RV and the true parameters')
xlabel('days')
ylabel('volatility level')
legend('True variance','Filtered variance','2.5%','97.5%','Location','northwest')
xlim([0,num_days])
nexttile
plot(V,'k')
hold on
plot(V_RV_est(:,1),'Color','b')
plot(V_RV_est(:,2),'Color' ,'r','LineStyle','--')
plot(V_RV_est(:,3),'Color' ,'r','LineStyle','--')
legend('True variance','Filtered variance','2.5%','97.5%','Location','northwest')
title('Volatility filtered using returns + RV and the estimated parameters')
xlim([0,num_days])
nexttile
plot(V,'k')
hold on
plot(V_RV_true(:,1),'Color' ,'r')
plot(V_RV_est(:,1),'Color','b')
legend('True variance','Filtered variance true parameters','Filtered Variance estimated parameters','Location','northwest')
xlim([0,num_days])
title('Volatility filtered using the true and estimated parameters')

%% Graph shape of the log-likelihood function for returns only

% Chose a set of parameters
mu = Opt_ret(1)/10;
kappa = exp(Opt_ret(2));
theta = exp(Opt_ret(3));
sigma = exp(Opt_ret(4));
rho   = (1-exp(-Opt_ret(5)))./(1+exp(-Opt_ret(5)));
x =[mu, kappa, theta, sigma , rho];

ret_mu = [linspace(0.06-0.2,0.06+0.2,50),0.06,x(1)];
ret_kappa = [linspace(5-3,5+3,50),5,x(2)];
ret_theta = [linspace(0.02-0.01,0.02+0.015,50),0.02,x(3)];
ret_sigma = [linspace(0.3-0.1,0.3+0.1,50),0.3,x(4)];
ret_rho = [linspace(-0.7-0.2,-0.7+0.2,50),-0.7,x(5)];

L_ret_mu = zeros(length(ret_mu),1);
L_ret_kappa = zeros(length(ret_kappa),1);
L_ret_theta = zeros(length(ret_theta),1);
L_ret_sigma = zeros(length(ret_sigma),1);
L_ret_rho = zeros(length(ret_rho),1);


for i=1:length(ret_mu)
    x1 = x;
    x1(1) = ret_mu(i);
    L_ret_mu(i,1) = Heston93_nat(x1,simul, ret,1);
    
    x1 = x;
    x1(2) = ret_kappa(i);
    L_ret_kappa(i,1) = Heston93_nat(x1,simul, ret,1);
    
    x1 = x;
    x1(3) = ret_theta(i);
    L_ret_theta(i,1) = Heston93_nat(x1,simul, ret,1);
    
    x1 = x;
    x1(4) = ret_sigma(i);
    L_ret_sigma(i,1) = Heston93_nat(x1,simul, ret,1);
    
    x1 = x;
    x1(5) = ret_rho(i);
    L_ret_rho(i,1) = Heston93_nat(x1,simul, ret,1);
    i
end

%%
figure
[ret_mu2,idx]=sort(ret_mu);
tiledlayout(3,2) 
nexttile
plot(ret_mu2,L_ret_mu(idx),'k')
hold on
plot(ret_mu(end-1),L_ret_mu(end-1),'r*')
plot(ret_mu(end),L_ret_mu(end),'b*')
ylabel('Log-Likelihood')
xlabel('\mu')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \mu')

nexttile
[ret_kappa2,idx]=sort(ret_kappa);
plot(ret_kappa2,L_ret_kappa(idx),'k')
hold on
plot(ret_kappa(end-1),L_ret_kappa(end-1),'r*')
plot(ret_kappa(end),L_ret_kappa(end),'b*')
ylabel('Log-Likelihood')
xlabel('\kappa')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \kappa')

[ret_theta2,idx]=sort(ret_theta);
nexttile
plot(ret_theta2,L_ret_theta(idx),'k')
hold on
plot(ret_theta(end-1),L_ret_theta(end-1),'r*')
plot(ret_theta(end),L_ret_theta(end),'b*')
ylabel('Log-Likelihood')
xlabel('\theta')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \theta')

[ret_sigma2,idx]=sort(ret_sigma);
nexttile
plot(ret_sigma2,L_ret_sigma(idx),'k')
hold on
plot(ret_sigma(end-1),L_ret_sigma(end-1),'r*')
plot(ret_sigma(end),L_ret_sigma(end),'b*')
ylabel('Log-Likelihood')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \sigma')

[ret_rho2,idx]=sort(ret_rho);
nexttile
plot(ret_rho2,L_ret_rho(idx),'k')
hold on
plot(ret_rho(end-1),L_ret_rho(end-1),'r*')
plot(ret_rho(end),L_ret_rho(end),'b*')
ylabel('Log-Likelihood')
xlabel('\rho')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \rho')


%% Graph shape of the log-likelihood function for returns and RV
mu = Opt_rv(1)/10;
kappa = exp(Opt_rv(2));
theta = exp(Opt_rv(3));
sigma = exp(Opt_rv(4));
rho   = (1-exp(-Opt_rv(5)))./(1+exp(-Opt_rv(5)));
eta = exp(Opt_rv(6));
x =[mu, kappa, theta, sigma , rho, eta];

rv_mu = [linspace(0.06-0.2,0.06+0.2,50),0.06,x(1)];
rv_kappa = [linspace(5-3,5+3,50),5,x(2)];
rv_theta = [linspace(0.02-0.01,0.02+0.015,50),0.02,x(3)];
rv_sigma = [linspace(0.3-0.1,0.3+0.1,50),0.3,x(4)];
rv_rho = [linspace(-0.7-0.2,-0.7+0.2,50),-0.7,x(5)];

L_rv_mu = zeros(length(rv_mu),1);
L_rv_kappa = zeros(length(rv_kappa),1);
L_rv_theta = zeros(length(rv_theta),1);
L_rv_sigma = zeros(length(rv_sigma),1);
L_rv_rho = zeros(length(rv_rho),1);


for i=1:length(rv_mu)
    x1 = x;
    x1(1) = rv_mu(i);
    L_rv_mu(i,1) = Heston93_nat(x1,simul, [ret, RV],1);
    
    x1 = x;
    x1(2) = rv_kappa(i);
    L_rv_kappa(i,1) = Heston93_nat(x1,simul, [ret, RV],1);
    
    x1 = x;
    x1(3) = rv_theta(i);
    L_rv_theta(i,1) = Heston93_nat(x1,simul, [ret, RV],1);
    
    x1 = x;
    x1(4) = rv_sigma(i);
    L_rv_sigma(i,1) = Heston93_nat(x1,simul, [ret, RV],1);
    
    x1 = x;
    x1(5) = rv_rho(i);
    L_rv_rho(i,1) = Heston93_nat(x1,simul, [ret, RV],1);
    i
end

%%
figure
[rv_mu2,idx]=sort(rv_mu);
tiledlayout(3,2) 
nexttile
plot(rv_mu2,L_rv_mu(idx),'k')
hold on
plot(rv_mu(end-1),L_rv_mu(end-1),'r*')
plot(rv_mu(end),L_rv_mu(end),'b*')
ylabel('Log-Likelihood')
xlabel('\mu')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \mu')

nexttile
[rv_kappa2,idx]=sort(rv_kappa);
plot(rv_kappa2,L_rv_kappa(idx),'k')
hold on
plot(rv_kappa(end-1),L_rv_kappa(end-1),'r*')
plot(rv_kappa(end),L_rv_kappa(end),'b*')
ylabel('Log-Likelihood')
xlabel('\kappa')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \kappa')

[rv_theta2,idx]=sort(rv_theta);
nexttile
plot(rv_theta2,L_rv_theta(idx),'k')
hold on
plot(rv_theta(end-1),L_rv_theta(end-1),'r*')
plot(rv_theta(end),L_rv_theta(end),'b*')
ylabel('Log-Likelihood')
xlabel('\theta')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \theta')

[rv_sigma2,idx]=sort(rv_sigma);
nexttile
plot(rv_sigma2,L_rv_sigma(idx),'k')
hold on
plot(rv_sigma(end-1),L_rv_sigma(end-1),'r*')
plot(rv_sigma(end),L_rv_sigma(end),'b*')
ylabel('Log-Likelihood')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \sigma')

[rv_rho2,idx]=sort(rv_rho);
nexttile
plot(rv_rho2,L_rv_rho(idx),'k')
hold on
plot(rv_rho(end-1),L_rv_rho(end-1),'r*')
plot(rv_rho(end),L_rv_rho(end),'b*')
ylabel('Log-Likelihood')
xlabel('\rho')
legend('Log-Likelihood','True parameter','Optimized parameter')
title('Parameter \rho')
