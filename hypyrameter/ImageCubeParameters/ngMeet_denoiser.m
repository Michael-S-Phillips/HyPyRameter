function [denoised_data] = ngMeet_denoiser(valid_cube, wavelengths, num_iter)
disp(sprintf('\tinstantiating hypercube object'))
hcube=hypercube(valid_cube,wavelengths)
%For parameter choices:
%We turn off 'NoiseWhiten' because of bad numerics in computed covariance
%matrices. (Get Inf in covariance matrix and can't get eigenvalues)
%The crism nan values throw the noiseWhiten routine off
disp(sprintf('\tcounting endmembers'))
numEndmembers = countEndmembersHFC(hcube, 'NoiseWhiten', true);
disp([9 num2str(numEndmembers) ' endmembers'])

%Thinking we should have a max value here- what should it be??
max_spectral_subspace= 12;

spec_subspace =  min([numEndmembers, max_spectral_subspace]);

disp(sprintf('\tstarting denoiseNGMeet'))
%Using the default for 'Sigma', which is .1*\sigma_n, the noise variance
denoised_cube = denoiseNGMeet(hcube, 'SpectralSubspace', spec_subspace, 'NumIterations', num_iter);


%Need to put the crism nan values back in where we had data issues.
denoised_data = denoised_cube.DataCube;


disp('MATLAB denoising completed.')
