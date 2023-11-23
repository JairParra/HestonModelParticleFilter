function [Like_H,V]=Heston93_nat(x,simul, Obs, C_I)
% The scritp uses the Malik and Pitt approach developped for the particle
% filter for the Heston model.
% Input:
%        x (6 x 1) : Transformed parameters     
%        simul : Structure containing the simulated noise
%        nb_part (1 x 1) : The number of partilce (ex : 5000)
%        Obs (n x 1) or (n x 2) : The observations. Can be a 1d-array if only
%                                 the log-returns are included or a 2-d array if the log-returns and
%                                 the Realized Variance are included
%                                 varargin (n x 1) : If Integrated Variance is used
%       C_I (1 x 1)             : Takes values 0 or 1. This bolean variable
%                                 is used to determine if the filtered
%                                 variance is computed.
%                                 N. B. C_I should be set to 0 during the
%                                 optimization process!!! Otherwise the
%                                 optmization will be much slower.
% Output:
%        [Like_H,V] : Negative Log likelikelihood, Filtered average
%        variance. If C_I==1 then V will be (n x 3) and will contain the
%        confidence 95% interval for the variance.
% Autor Rémi Galarneau-Vincent
    size_obs = size(Obs,2);

    % Number of observations
    N = length(Obs);
    
    % Uniform rv used for propagating the partilces
    U = simul.U;
    
    ZZ = simul.ZZ;
    % Uniform rv used for resampling the particles
    UU = simul.UU;
    
    
    % Inverse Gaussian rv used for sampling the Integrated variance
    %U_IV = simul.U_IV;
    
    % If the realized variance is given
    if size_obs==2
        Y  = Obs(:,1);
        RV = Obs(:,2);
        eta_p   = x(6);
    else
        Y  = Obs(:,1);
    end
    if nargin==3
        C_I=0;
    end
    % Transform the parameters to their original values
    mu    = x(1);
    kappa = x(2);
    theta = x(3);
    sigma = x(4);
    rho   = x(5);

    % Time step
    delta = 1./252;

    % Initial sampling of the volatility component.
    v1 = (U(:,1).*0.02);
    
    % Use the inverse method to generate Non-central chi-squared rv from
    % uniform rv

    % Pre-allocate the vector of filtered variance
    if C_I==0
        V = zeros(N,1);
    else
        V = zeros(N,3);
    end
    
    % Pre-allocate the vector for the log-likelihood
    Like_H = zeros(N,1);


    % Select the probability density function based on the observation set.
    if size_obs==1
        pdf = @(Y, IV_s, v1, v2) normpdf(Y,mu.*delta - 0.5.*IV_s... 
                     + rho./sigma.*(v2-v1-kappa.*theta.*delta...
                     +kappa.*IV_s),sqrt((1-rho.^2).*IV_s));
    else
        pdf = @(Y,RV, IV_s, v1, v2) normpdf(Y,mu.*delta - 0.5.*IV_s ...
                     + rho./sigma.*(v2-v1-kappa.*theta.*delta...
                     +kappa.*IV_s),sqrt((1-rho.^2).*IV_s))...
                     .*normpdf((IV_s- RV)./RV,0, eta_p);   
    end


    for t = 1:N
        % I use the method proposed in Glasserman to sample quickly
        % non-central chi-squared rv.
        % I simulate a one time propagation step v_t to v_t+1 
        IV = 0;
        vol_past = v1;
        for d=1:5
            v2 = max(vol_past + kappa.*(theta-vol_past).*delta/5 + sigma.*sqrt(vol_past.*delta/5).*ZZ(:,t,d),0.000001);
            IV = IV+ (vol_past+v2)/2.*delta./5;
            vol_past = v2;
        end
        % The log-likelihood. Its important to note that since the
        % returns and the volatility are correlated we have to take
        % it into account.
        
        if size_obs==1
            w = pdf(Y(t), IV, v1, v2);
        else
            w = pdf(Y(t),RV(t), IV, v1, v2);
        end
        
        % I compute the Log-Likelihood            
        Like_H(t) = log(mean(w));

        % I compute each particles weight
        pis = w./sum(w);
        

        % I use the Malik & Pitt 2011 method to smooth the volatility CDF
        % Concatenate the volatility and their respective weight in an Nx2
        % matrix.
        
        % Note that I adapted the Malik and Pitt method to run quickly
        
        arr = [v2,pis];
        arr = arr(pis>10^-7,:);

        % Sort in ascending order the matrix based on the first column
        % that contains the volatility values
        arr = sortrows(arr,1);
        
        % Compute the empirical CDF
        arr(:,2) = cumsum(arr(:,2))-arr(:,2)./2;
        
        % I compute the expected volatility value and confidence interval.
        
        % Complete the CDF for the smoothing process
        arr = [[arr(1,1)*0.95,0]; arr; [arr(end,1)*1.05,1]];
        if C_I==1
            V(t,1) = sum(v2.*pis);
            V(t,2) = interp1(arr(:,2),arr(:,1),0.025);
            V(t,3) = interp1(arr(:,2),arr(:,1),0.975);
        end
        % Use linear interpolation to apply Malik & Pitt smoothing
        % method to stabilize the Likelihood

        v1 = interp1(arr(:,2),arr(:,1),UU(:,t));
        
    end
    Like_H = sum(-Like_H);

     
end


