function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% COST:

% Hypothesis: h_theta
h_theta = sigmoid(X*theta);

% RegularizedTheta: (excluding theta_0)
regularized_theta = theta(2:end, :);

% RegularizationParameter: = (lambda / (2 * m)) * Sum_j(1,n) theta^2   [Sum_j(1,n) theta^2 = theta'*theta]
regularization_param = lambda/(2*m) * (regularized_theta' * regularized_theta);  

% Cost: J
cost = - (1/m) * ((y' * log(h_theta)) + ((1-y')*log(1-h_theta)));
J = cost + regularization_param;



% GRADIENT:

% PartialDerivative: compute the usual partial derivatives, without regularization
partial_derivative = 1/m * (X' * (h_theta - y));

% RegularizationParameter:
partial_derivative_regularization_param = (lambda/m)*theta;

% RegularizationParameter0: apply regularization, except for theta = 1 so set first term = 0
partial_derivative_regularization_param(1) = 0;

% Gradient:
grad = partial_derivative + partial_derivative_regularization_param;

% =============================================================

end
