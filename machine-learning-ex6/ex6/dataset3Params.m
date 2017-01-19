function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_array = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_array = [0.01 0.03 0.1 0.3 1 3 10 30];

clen = length(C_array);
slen = length(sigma_array);

error = zeros(clen, slen);

x1 = X(:,1);
x2 = X(:,2);

for i = 1:length(C_array)
    for j = 1:length(sigma_array)
        model = svmTrain(X, y, C_array(i), @(x1, x2) gaussianKernel(x1, x2, sigma_array(j)));
        predictions = svmPredict(model, Xval);
        error(i,j) = mean(double(predictions ~= yval));
    end
end

min_err = min(min(error));

idx =  find(ismember(error , min_err ))(1);

[r,c] = ind2sub(size(error), idx);

C = C_array(r);
sigma = sigma_array(c);

% =========================================================================

end
