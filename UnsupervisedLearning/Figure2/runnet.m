function [rO, O, V, ii, ie] = runnet(dt, lambda, F ,Input, C,Nneuron,Ntime, Thresh, trackCurrents)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%% This function runs the network without learning. It take as an
%%%% argument the time step dt, the leak of the membrane potential lambda,
%%%% the Input of the network, the recurrent connectivity matrix C, the feedforward
%%%% connectivity matrix F, the number of neurons Nneuron, the length of
%%%% the Input Ntime, and the Threhsold. It returns the spike trains O
%%%% the filterd spike trains rO, and the membrane potentials V.
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rO=zeros(Nneuron,Ntime);%filtered spike trains
O=zeros(Nneuron,Ntime); %spike trains array
V=zeros(Nneuron,Ntime); %membrane potential array
ie=zeros(Nneuron,Ntime); %membrane potential array
ii=zeros(Nneuron,Ntime); %membrane potential array

ie_test=zeros(Nneuron,Ntime);
ii_test=zeros(Nneuron,Ntime);

for t=2:Ntime
    
    if trackCurrents
        for n=1:Nneuron
            n_in = [dt*(F(:,n).*Input(:,t-1)); C(n,:)*O(:,t-1)];
            ie(n,t) = sum(n_in(n_in>0));
            ii(n,t) = -sum(n_in(n_in<0));

            n_in = C(n,:)*O(:,t-1);
            ie_test(n,t) = sum(n_in(n_in>0));
            ii_test(n,t) = sum(n_in(n_in<0));
        end
    end

    V(:,t)=(1-lambda*dt)*V(:,t-1)+dt*F'*Input(:,t-1)+C*O(:,t-1)+0.001*randn(Nneuron,1);%the membrane potential is a leaky integration of the feedforward input and the spikes

     % Find neuron with largest membrane potential & update weights and spike train accordingly
    [m,k]= max(V(:,t) - Thresh-0.01*randn(Nneuron,1));
    if (m>=0) % spike if largest V over threshold
        O(k,t)=1;
    end

    rO(:,t)=(1-lambda*dt)*rO(:,t-1)+1*O(:,t); %filtering the spikes
end

end




