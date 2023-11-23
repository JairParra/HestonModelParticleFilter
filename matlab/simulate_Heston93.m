function [ret, V, IV] = simulate_Heston93(mu, kappa, theta, sigma, rho, v_start, num_days)
%Autor Rémi Galarneau-Vincent
    
    
    % Check if the parameters are valid
    k = 0;
    if theta<=0
        warning('Theta should be greater than 0')
        k = 1;
    end

    if sigma<=0
        warning('Sigma should be greater than 0')
        k = 1;
    end
    if kappa<=0
        warning('kappa should be greater than 0')
        k = 1;
    end
    if (2*kappa*sigma)<=(sigma^2)
        warning('2*kappa*theta should be greater than sigma^2')
        k = 1;
    end

    if k ==1
        warning('Stopping')
    else
        
        % time step, I use one trading day
        delta = 1/252;

        % the d and eta are paramters used for simulating non-central chisquare 
        % distributions
%         df = 4*kappa*theta/sigma^2;
%         eta   = 4*kappa*exp(-kappa*delta)/(sigma^2*(1-exp(-kappa*delta)));
%         alpha = exp(-kappa*delta)/eta;

        % preallocate the array that will store the volatility
        V = zeros(num_days+1,1);
        %V(1) = v;

        % preallocate the array that will store the returns
        ret = zeros(num_days,1);
        IV = zeros(num_days,1);
        v = zeros(5,1);        
        for t = 1:num_days
            if t==1
                vol_past = v_start;
            else
                vol_past = V(t-1);
            end
            v(1,1) = max(vol_past + kappa*(theta-vol_past)*delta/5 + sigma.*sqrt(vol_past*delta/5)*normrnd(0,1),0.000001);
            for d=2:5
                v(d,1) = max(v(d-1) + kappa*(theta-v(d-1))*delta/5 + sigma.*sqrt(v(d-1)*delta/5)*normrnd(0,1),0.000001);
            end
            vol = [vol_past;v];
            V(t) = v(5,1);
            IV(t) = sum((vol(1:end-1)+vol(2:end))/2)*delta/5;
            ret(t) = mu*delta -IV(t)/2 + rho/sigma*(V(t)-vol_past-kappa*theta*delta+...
                    kappa*IV(t)) + normrnd(0,sqrt((1-rho.^2)*IV(t)));
            IV(t) = IV(t) + normrnd(0,0.1.*IV(t));
        end
    
    end



end