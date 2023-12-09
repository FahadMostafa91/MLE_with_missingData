
%Generate synthetic data with a least squares model.
%Introduce missing data.
%Implement the MLE to estimate parameters.
%Here's an example MATLAB code to demonstrate this process:
% Step 1: Generate synthetic data
rng(42); % Set random seed for reproducibility

% True parameters of the model
true_slope = 1.5;
true_intercept = 10;

% Generate synthetic data
time_in_weeks = 1:50;
true_cases = true_slope * time_in_weeks + true_intercept;

% Add noise to the data
noise = 5 * randn(size(time_in_weeks));
observed_cases = true_cases + noise;

% Step 2: Introduce missing data
missing_data_percentage = 0.2; % 20% missing data
num_missing_points = round(missing_data_percentage * length(time_in_weeks));

% Randomly select points to be missing
missing_indices = randperm(length(time_in_weeks), num_missing_points);

% Create a copy of the observed cases with missing values
missing_data_cases = observed_cases;
missing_data_cases(missing_indices) = NaN;

% Step 3: Implement MLE to estimate parameters
% Define the likelihood function for the least squares model
likelihood_function = @(params) sum((observed_cases - params(1) * time_in_weeks - params(2)).^2);

% Use fminsearch to find parameters that maximize the likelihood
initial_guess = [1, 5]; % Initial guess for the slope and intercept
estimated_params = fminsearch(likelihood_function, initial_guess);

% Display true and estimated parameters
disp('True Parameters:');
disp(['True Slope: ' num2str(true_slope)]);
disp(['True Intercept: ' num2str(true_intercept)]);
disp(' ');

disp('Estimated Parameters using MLE:');
disp(['Estimated Slope: ' num2str(estimated_params(1))]);
disp(['Estimated Intercept: ' num2str(estimated_params(2))]);

% Plot the synthetic data
figure;
plot(time_in_weeks, observed_cases, 'o', 'DisplayName', 'Observed Data');
hold on;
plot(time_in_weeks, true_cases, '--', 'LineWidth', 2, 'DisplayName', 'True Model');
plot(time_in_weeks, estimated_params(1) * time_in_weeks + estimated_params(2), '-', 'LineWidth', 2, 'DisplayName', 'Estimated Model');
scatter(time_in_weeks(missing_indices), missing_data_cases(missing_indices), 50, 'rx', 'DisplayName', 'Missing Data');
xlabel('Time (Weeks)');
ylabel('Number of Cases');
title('Synthetic Data with Missing Values and MLE Fit');
legend('Location', 'Best');
grid on;
hold off;

%Maximum Likelihood Estimation (MLE) is a method used to estimate the parameters of a statistical model by maximizing the likelihood function. However, when dealing with missing data, the likelihood function needs to account for the missing values. This often involves considering the joint likelihood of the observed data and the missing data.

%One common approach is to use the Expectation-Maximization (EM) algorithm, which iteratively estimates missing data and updates the parameter estimates. The EM algorithm alternates between an E-step (Expectation) and an M-step (Maximization) until convergence.

%Here's a simplified explanation of how EM works in the context of missing data:

%Expectation (E-step): Compute the expected value of the missing data given the observed data and current parameter estimates.
%Maximization (M-step): Maximize the likelihood function based on both the observed and imputed (estimated) data, updating the parameter estimates.
%Repeat the E-step and M-step until convergence.
%Implementing EM can be more involved than the previous example I provided for complete data. Below is a simple MATLAB example using the EM algorithm to estimate parameters for a linear regression model with missing data:


rng(42); % Set random seed for reproducibility

% True parameters of the model
true_slope = 1.5;
true_intercept = 10;

% Generate synthetic data
time_in_weeks = 1:50;
true_cases = true_slope * time_in_weeks + true_intercept;

% Add noise to the data
noise = 5 * randn(size(time_in_weeks));
observed_cases = true_cases + noise;

% Introduce missing data
missing_data_percentage = 0.2; % 20% missing data
num_missing_points = round(missing_data_percentage * length(time_in_weeks));
missing_indices = randperm(length(time_in_weeks), num_missing_points);
observed_cases_with_missing = observed_cases;
observed_cases_with_missing(missing_indices) = NaN;

% Initialize parameters
initial_guess = [1, 5]; % Initial guess for the slope and intercept
current_params = initial_guess;

% EM Algorithm
max_iter = 100;
tolerance = 1e-6;
iter = 1;

while iter <= max_iter
    % E-step: Impute missing data using current parameter estimates
    imputed_data = current_params(1) * time_in_weeks + current_params(2);
    observed_cases_with_missing(missing_indices) = imputed_data(missing_indices);
    
    % M-step: Maximize the likelihood function and update parameter estimates
    updated_params = fminsearch(@(params) sum((observed_cases_with_missing - params(1) * time_in_weeks - params(2)).^2), current_params);
    
    % Check for convergence
    if norm(updated_params - current_params) < tolerance
        break;
    end
    
    % Update parameters for the next iteration
    current_params = updated_params;
    iter = iter + 1;
end

% Display true and estimated parameters
disp('True Parameters:');
disp(['True Slope: ' num2str(true_slope)]);
disp(['True Intercept: ' num2str(true_intercept)]);
disp(' ');

disp('Estimated Parameters using EM:');
disp(['Estimated Slope: ' num2str(current_params(1))]);
disp(['Estimated Intercept: ' num2str(current_params(2))]);
