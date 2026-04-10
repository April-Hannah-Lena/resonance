addpath('/Users/april/Documents/MATLAB/chebfun/');
addpath('/Users/april/Documents/MATLAB/REfit/');

tol = 1e-7;

autocorrelation = readmatrix('autocorrelation.txt');
%autocorrelation_real = readmatrix('autocorrelation_real.txt');
%autocorrelation_imag = readmatrix('autocorrelation_imag.txt') * 1i;
%autocorrelation = autocorrelation_real + autocorrelation_imag;
%n = round(length(autocorrelation) / 2, TieBreaker='tozero');
n = length(autocorrelation);

%rho = @(z) z.^(0:-1:-(n-1)) * autocorrelation(1:n); %+ z.^(1:1:n) * conj(autocorrelation(2:end)); %(z.^(0:n)) * corr; %                % power spectrum (spectral measure)
%rho = @(z) z.^(-n:n) * autocorrelation;
%tpts = transpose(linspace(0,1,10000));
%zpts = exp(2*pi*1i*tpts);

%R = rfun(rho(zpts),tpts);
S = efun(autocorrelation,0:n-1,'coeffs','tol',tol);%-n:n,'coeffs');
R = rfun(S);

poleR = poles(R,'zt');
%poleR = poleR(abs(poleR)<=1);
rootR = roots(R,'zt');
%rootR = rootR(abs(rootR)<=1);
resR = residues(R);
%poleS = poles(S,'zt');
%poleS = poleS(abs(poleS)<=1);
%rootS = roots(S,'zt');
%rootS = rootS(abs(rootS)<=1);

writematrix(poleR,'poles.txt');
writematrix(rootR,'roots.txt');
writematrix(resR,'residues.txt');
%writematrix(poleS,'poles.txt');
%writematrix(rootS,'roots.txt');
%path()