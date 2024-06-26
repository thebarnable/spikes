% This program produces Figure 1C for "Optimal Compensation for
% Neuron Loss" by David Barrett, Sophie Deneve, and Christian Machens

% This program simulates a homogeneous network of spiking neurons. 
% All free parameters can be changed from their default values 
% (within limits, since plotting is not optimized for large changes)

rng('default');

%--------------------------- FREE PARAMETERS -------------------------------

N    = 20;                       % Number of neurons in network
Nko  = 0;                        % indices of neurons to be k.o.
lamd = 10;                      % Decoder timescale (in inverse milliseconds)
lams = 5;
lamv = 10;
QBeta = 0.05;                    % Quadratic firing rate cost
sigV = 0;                    % standard deviation of voltage noise
D = ones(1,N);                 % Decoder (homogeneous by default)
% D = randn(1,N);    

%--------------------------- SIMULATION AND DERIVED PARAMETERS -------------

% Connectivity
QBeta = QBeta / N^2;             % scale costs according to network size
Om=D'*D;% + QBeta*eye(N);        % Initialise Recurrent connectivity
Os=zeros(N,N);
Os=(D'*(-lams+lamd)*D); % Initialise Recurrent connectivity
% Os=Os-diag(diag(Os));
T=diag(Om)/2;                    % thresholds

% Time steps (Euler method)
Time = 2;                      % Simulation time in seconds
tko  = 60;                       % time of knockout
dt = 0.0001;                       % Time steps in seconds
t = 0:dt:Time-dt;                % array of time points
Nt=length(t);                    % Number of time steps

% Input signal
old=false;
if old
    xsignal = 300;
    x=zeros(1,Nt);                   % Initialise Signal ...
    x(Nt/8+1:(Nt-2)/3)=xsignal/200;
    x((Nt-2)/3+1:Nt)=xsignal/100;
    x = smooth( x, Nt/50 );          % smooth away the steps
    dxdt = [0,diff(x)]/dt;           % Compute signal derivative
    c = lams*x + dxdt;             % actual input into network
else
    c = zeros(1, Nt);
    c(:, 1:2000) = 0;
    c(:, 2000:5000) = 10;
    c(:, 5000:7000) = 0;
    c(:, 7000:9000) = 5;
    c(:, 9000:Nt) = 0;
    x=zeros(1, Nt); % the target output/input
    for ti=2:Nt
        x(:,ti)= (1-lams*dt)*x(:,ti-1)+ dt*c(:,ti-1);
    end
end
%--------------------------  SIMULATION ------------------------------------
% initial conditions
V = zeros(N,Nt);                 % voltages
s = zeros(N,Nt);                 % Spike trains
r = zeros(N,Nt);                 % Filtered spike trains (or firing rates)

ie=zeros(N,Nt); %membrane potential array
ii=zeros(N,Nt); %membrane potential array

% Simulate network
trackCurrents=true;
for k=2:Nt
    if trackCurrents
        for n=1:N
            n_in = [dt*(D'*c(:,k-1)); Os*r(:,k-1)*dt; - Om*s(:,k-1)*dt];
            ie(n,k) = sum(n_in(n_in>0));
            ii(n,k) = -sum(n_in(n_in<0));
        end
    end
  
  % Voltage and firing rate update with Euler method
  dVdt     = -lamv*V(:,k-1) + D'*c(:,k-1) + Os*r(:,k-1) - Om*s(:,k-1);
  drdt     = -lamd*r(:,k-1) + s(:,k-1);
  V(:,k)   = V(:,k-1) + dVdt*dt + sigV*randn(N,1).*sqrt(dt); 
  r(:,k)   = r(:,k-1) + drdt*dt;
  
  % knock-out neuron after time point 'tko'
  %if t(k)>tko, V(Nko,k) = 0; end;
  
  % check threshold crossings; only one neuron should spike per
  % time step (this is a numerical trick which allows us to use
  % a larger time step; in general, the network becomes sensitive 
  % to delays if the redundancy grows, and the input dimensionality
  % remains small and fixed)
  spiker  = find( V(:,k) > T);
  Nspiker = length(spiker);
  if Nspiker>0
    chosen_to_spike=spiker(randi(Nspiker)); 
    s(chosen_to_spike,k)=1/dt;
  end

end
xest = D*r;                      % compute readout with original decoder


% Define the cutoff frequency (Hz) and sample rate (Hz)
NeuronToPlot=5;
cutoffFrequency = 50; % Example cutoff frequency
sampleRate = 10000; % Example sample rate
ii = ii(NeuronToPlot,:);
ie = ie(NeuronToPlot,:);

% Apply lowpass filter
ii = lowpass(ii, cutoffFrequency, sampleRate);
ie = lowpass(ie, cutoffFrequency, sampleRate);
% Create a new figure (but do not display it)
fig = figure('Visible', 'off'); % 'Visible', 'off' makes the figure invisible

% Plot the first vector
tii=1:length(ii);
plot(tii, ii, 'r') % 'r' specifies a red line
hold on % Hold the plot to add another line

% Plot the second vector
plot(tii, ie, 'b') % 'b' specifies a blue line

% Optional: add labels, title, and legend
xlabel('X Axis')
ylabel('Y Axis')
title('Plot of ii and ie over the same X Axis')
legend('ii', 'ie')

% Save the plot to a file
saveas(fig, "currents_lp"+cutoffFrequency+".png") % Save as PNG file
% or use print
% print(fig, 'plot_ii_ie', '-dpng') % Save as PNG file

% Close the figure
close(fig)

%=========================== FIGURE 1C =====================================

figure(1); clf;
set(gcf, 'Color', 'w');

TimeEnd = Time;               % Right range of plots

% plot stimulus and estimate
axes('pos',[0.1 0.6 0.8 0.35 ] );
hold on;
plot( t, x,   '-k','LineWidth', 1.5);
plot( t, xest,'-','LineWidth', 1.5);
set( gca,'XTick',[], 'XColor','w');
set( gca,'YTick', [0 2 4],'YTickLabel', {'0', '', '4'}, 'TickDir', 'out' );
set( gca, 'LineWidth', 0.75);
ylabel( 'Signal (a.u.)');
axis( [0 TimeEnd -0.1 4 ]);

h = 0.32/N;
hs = 0.4/N;
v = 0.1:hs:(0.1+(N-1)*hs);

% plot voltage traces
for k=1:N
  axes('pos',[0.1 v(k) 0.8 h ] );
  spikesize=1.6*T(k);
  plot( t, V(k,:)+s(k,:)*spikesize*dt,'-k','LineWidth', 1.5);
  axis([0 TimeEnd -T(k) T(k)+spikesize/0.8]);
  text(-4,-0.1, sprintf( 'V_{%d}', k ) );
  axis off;
end
pos = get( gca, 'pos' );
scl = pos(3)/TimeEnd;
scalebar([0.9-scl*15, 0.08, scl*10, 0.01], '10 msec');

