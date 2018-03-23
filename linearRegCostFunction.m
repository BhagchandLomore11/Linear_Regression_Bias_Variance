function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

grad = zeros(size(theta));

predictions = X*theta; % predictions of hypothesis example

square_error = (predictions - y).^2; % squared error

J = (1/(2*m))*sum(square_error); % value of cost function w/o regularization

reg_term = (lambda/(2 * m)) * sum(theta(2:end).^2); % calculate regularization term

J = J + reg_term; % calculate cost with regularization

grad = X'*(predictions - y) / m + lambda*[0;theta(2:end)] / m; % calculate gradient with regularization

end
