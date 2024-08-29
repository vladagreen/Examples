library(tidyverse)
library(ggplot2)
library(caret)
library(glmnet)
library(caTools)
library(dplyr)
library(rsample)  
library(recipes)
library(themis)
library(yardstick)
library(tidymodels)
library(ranger)  
library(workflows)
library(vip)
library(readr)
library(fastDummies)
library(stats)
library(Metrics)



data <- read_csv("/Users/vladasliusar/d/new_york_listings_2024.csv")
full_data <- read_csv("/Users/vladasliusar/d/new_york_listings_2024.csv")

#DATA PREPARATION

# Drop the variables 'id', 'name', 'host_id', 'host_name'
data <- select(data, -c(id, name, host_id, host_name, neighbourhood, last_review, calculated_host_listings_count))

# there are no missing values
has_missing_values <- any(is.na(data))

# Print the result
if (has_missing_values) {
  print("There are missing values in the dataset.")
} else {
  print("There are no missing values in the dataset.")
}

# Replace 'Studio' with 0 in the 'bedrooms' column
data$bedrooms[data$bedrooms == "Studio"] <- 0

# Remove rows where 'baths' is 'Not specified'
data <- data[data$baths != 'Not specified', ]

#make the variable 'license' have 3 categories
data <- data %>%
  mutate(license = case_when(
    license == "Exempt" ~ "Exempt",
    license == "No License" ~ "No License",
    TRUE ~ "License"  # This line covers all other cases
  ))

# Convert 'baths' and 'beds' to numeric in the original dataset
data$baths <- as.numeric(as.character(data$baths))
data$beds <- as.numeric(as.character(data$beds))
data$bedrooms <- as.numeric(as.character(data$bedrooms))

# Convert categorical features to factors in data
data$neighbourhood_group <- as.factor(data$neighbourhood_group)
data$room_type <- as.factor(data$room_type)
data$license <- as.factor(data$license)


# Replace 'No rating' with the mean rating
data$rating <- as.numeric(replace(data$rating, data$rating == "No rating", NA))
mean_rating <- mean(data$rating, na.rm = TRUE)
data$rating[is.na(data$rating)] <- mean_rating


#remove outliers from the data set
data <- data %>% 
  filter(price <= 5000) %>%
  filter(baths <= 2) %>% 
  filter(beds <= 4) %>%
  filter(number_of_reviews_ltm <= 60) 
  

# Now check the levels
print(levels(data$neighbourhood_group))
print(levels(data$room_type))
print(levels(data$license))


#Training test split
# Set seed for reproducibility
set.seed(123)
# Split the data without stratification
data_split <- initial_split(data, prop = 0.75)
# Create the training and test datasets
train_data <- training(data_split)
test_data <- testing(data_split)



set.seed(123)
# Create 10-fold cross-validation splits
data_folds <- vfold_cv(data, v = 10)

# Create the initial recipe
data_rec <- recipe(price ~ ., data = train_data) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())

# Define a set of regression metrics
regression_metrics <- metric_set(yardstick::rmse, yardstick::rsq, yardstick::mae)


# Now check the levels
print(levels(train_data$neighbourhood_group))
print(levels(train_data$room_type))
print(levels(train_data$license))


#MODELS

#Random forest----------------------------------------------------------------------------------------------
# Define the model specifications for Random Forest
rf_spec <- rand_forest(trees = 1000) %>%  # You can adjust the number of trees
  set_engine("ranger", importance = 'permutation') %>%
  set_mode("regression")

# Combine the preprocessing recipe and the model specifications into a workflow
rf_wf <- workflow() %>%
  add_recipe(data_rec) %>%
  add_model(rf_spec)

# Use cross-validation to evaluate the model
set.seed(123)
rf_res <- fit_resamples(
  rf_wf,
  data_folds,
  metrics = regression_metrics,
  control = control_resamples(save_pred = TRUE)
)

# Print the results
print(rf_res$.metrics)
#------------------------------------------------------------------------------------------------------------

#MODEL PREDICTION

# Now check the levels
print(levels(train_data$neighbourhood_group))
print(levels(train_data$room_type))
print(levels(train_data$license))

new_observation <- tibble(
  neighbourhood_group = "Manhattan",
  room_type = "Private room",
  license = "License",
  latitude = 40.7128,
  longitude = -74.0060,
  minimum_nights = 30,
  number_of_reviews = 45,
  reviews_per_month = 0.5,
  availability_365 = 20,
  number_of_reviews_ltm = 10,
  rating = 4.5,
  bedrooms = 1,
  beds = 2,
  baths = 1
)

# Adjust factor levels in new_observation to match training data
new_observation$neighbourhood_group <- factor(new_observation$neighbourhood_group, levels = levels(train_data$neighbourhood_group))
new_observation$room_type <- factor(new_observation$room_type, levels = levels(train_data$room_type))
new_observation$license <- factor(new_observation$license, levels = levels(train_data$license))

# Train the model on the full training dataset
final_rf_model <- fit(rf_wf, train_data)

# Make a prediction using the trained model
predicted_price <- predict(final_rf_model, new_observation)
print(predicted_price)

#Feature importance ------

# Extract the fitted model object from the final_rf_model object
rf_predictor <- final_rf_model %>%
  pull_workflow_fit() 

# Generate variable importance
vi <- vip(rf_predictor, num_features = ncol(train_data))

# Create a data frame from the vip object for plotting
vi_df <- vi$data

# Create the plot using the data frame
vi_plot <- ggplot(vi_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "Features", y = "Importance", title = "Feature Importance")

print(vi_plot)

# Extract the top variables based on importance
top_vars <- vi_df %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 4) %>%
  pull(Variable)

print(top_vars)

saveRDS(final_rf_model, file = "final_rf_model.rds")

rf_fitted <- pull_workflow_fit(final_rf_model)
rf_fitted$fit$ntree
rf_fitted$fit$err.rate[ntree(rf_fitted$fit), "OOB"]

fitted_rf_model <- final_rf_model %>%
  extract_fit_parsnip()
number_of_trees <- fitted_rf_model$fit$ntree
mtry <- fitted_rf_model$fit$mtry
oob_error_rate <- fitted_rf_model$fit$err.rate

# Predict on the test data and ensure it is numeric
test_predictions <- predict(final_rf_model, test_data, type = "numeric")

# Extract the true outcomes
true_outcomes <- test_data$price

# Create a data frame containing both the true outcomes and the predictions
results_df <- data.frame(
  truth = true_outcomes,
  estimate = test_predictions
)

# Calculate RMSE using the yardstick function
test_rmse <- yardstick::rmse(results_df, truth = truth, estimate = .pred)
print(test_rmse$.estimate)

# Calculate R^2 using the yardstick function
test_rsquared <- yardstick::rsq(results_df, truth = truth, estimate = .pred)
print(test_rsquared$.estimate)

# Calculate MAE using the yardstick function
test_mae <- yardstick::mae(results_df, truth = truth, estimate = .pred)
print(test_mae$.estimate)

test_mae$.estimate

rmse_val <- results_df() %>%
  rmse(truth = truth, .pred = .pred)
sprintf(rmse_val$.estimate)


