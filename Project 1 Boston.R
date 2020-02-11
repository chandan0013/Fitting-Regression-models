# Linear,Polynomial,Random Forest,Ridge,Lasso Regression with Cross 
# Validation for Boston Housing Data Set

# Adding the required libraries


require(boot)
require(caret)
require(cvTools)
require(DAAG)
require(datasets)
require(ggplot2)
require(graphics)
require(grDevices)
require(hydroGOF)
require(lattice)
require(MASS)
require(Matrix)
require(methods)
require(randomForest)
require(robustbase)
require(sp)
require(stats)
require(utils)
require(zoo)
require(glmnet)



attach(Boston); # dataset

# Model training to be used in poly regression without CV

smp_size <- floor(0.75 * nrow(Boston));
set.seed(123);
train_ind <- sample(seq_len(nrow(Boston)), size = smp_size);

train <- Boston[train_ind, ];
test <- Boston[-train_ind, ];

# Linear Regression
regression_form = formula(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat); # Formula for the linear regression model
medv_fit = lm(regression_form, data = Boston); # Linear Fit
medv_CVlm_fit = CVlm(data = Boston, form.lm = medv_fit, m=10,printit = TRUE); # Linear Fit with 10-fold cross validation
rmse_lin_fit = rmspe(medv_CVlm_fit$medv,medv_CVlm_fit$cvpred); #RMSE

# Linear Regression with significant variables
regression_form_mod = formula(medv ~ crim+zn+indus+chas+rm+dis+tax+ptratio+black+lstat);
medv_fit_mod = lm(regression_form_mod, data = Boston);
medv_CVlm_fit_mod = CVlm(data = Boston, form.lm = medv_fit_mod, m=10,printit = FALSE);
rmse_lin_fit_mod = rmspe(medv_CVlm_fit_mod$medv,medv_CVlm_fit_mod$cvpred); #RMSE

# Linear Regression
regression_form_fixed = formula(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat);
medv_fit_fixed = lm(regression_form_fixed,data = train);
medv_predict_fixed = predict.lm(medv_fit_fixed,test);
rmse_fixed = rmspe(test$medv, medv_predict_fixed);

#Quadratic Regression
regression_form2 = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)+(crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2); # Formula for the quadratic regression model
medv_fit2 = lm(regression_form2,data = Boston); # Quadratic fit
medv_CVlm_fit2 = CVlm(data = Boston, form.lm = medv_fit2, m=10,printit = FALSE);# Quadratic Fit with 10-fold cross validation
rmse_fit2 = rmspe(medv_CVlm_fit2$medv,medv_CVlm_fit2$cvpred); #RMSE

#Quadratic Regression with fixed training and data
regression_form2_fixed = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)+(crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2);
medv_fit2_fixed = lm(regression_form2_fixed,data = train);
medv_predict2_fixed = predict.lm(medv_fit2_fixed,test);
rmse_fixed2 = rmspe(test$medv, medv_predict2_fixed);

#Cubic Regression
regression_form3 = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^3)); # Cubic Formula for the model
medv_fit3 = lm(regression_form3,data = Boston); # Cubic Fit
medv_CVlm_fit3 = CVlm(data = Boston, form.lm = medv_fit3, m=10,printit = FALSE); # Cubic Fit with 10-fold cross validation
rmse_fit3 = rmspe(medv_CVlm_fit3$medv,medv_CVlm_fit3$cvpred); #RMSE

#Cubic Regression with fixed training and data
regression_form3_fixed = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^3));
medv_fit3_fixed = lm(regression_form3_fixed,data = train);
medv_predict3_fixed = predict.lm(medv_fit3_fixed,test);
rmse_fixed3 = rmspe(test$medv, medv_predict3_fixed);

#4th order polynomial regression

regression_form4 = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^3)+ I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^4)); # degree 4 Formula for the model
medv_fit4 = lm(regression_form4,data = Boston); # degree 4
medv_CVlm_fit4 = CVlm(data = Boston, form.lm = medv_fit4, m=10,printit = FALSE); # degree 4 with 10-fold cross validation
rmse_fit4 = rmspe(medv_CVlm_fit4$medv,medv_CVlm_fit4$cvpred); #RMSE

# 4th order polynomial regression with fixed training and data
regression_form4_fixed = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^3)+ I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^4));
medv_fit4_fixed = lm(regression_form4_fixed,data = train);
medv_predict4_fixed = predict.lm(medv_fit4_fixed,test);
rmse_fixed4 = rmspe(test$medv, medv_predict4_fixed);

#5th order polynomial regression

regression_form5 = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^3)+ I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^4)+I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^5)); # degree 5 Formula for the model
medv_fit5 = lm(regression_form5,data = Boston); # degree 5
medv_CVlm_fit5 = CVlm(data = Boston, form.lm = medv_fit5, m=10,printit = FALSE); # degree 5 Fit with 10-fold cross validation
rmse_fit5 = rmspe(medv_CVlm_fit5$medv,medv_CVlm_fit5$cvpred); #RMSE

#5th order polynomial regression for fixed training and test dataset
regression_form5_fixed = formula(medv ~ (crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^2) + I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^3)+ I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^4)+I((crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+black+lstat)^5));
medv_fit5_fixed = lm(regression_form5_fixed,data = train);
medv_predict5_fixed = predict.lm(medv_fit5_fixed,test);
rmse_fixed5 = rmspe(test$medv, medv_predict5_fixed);

# RMSE aginst poly degree plot

degree = c(1,2,3,4,5);
rmse = c(rmse_lin_fit,rmse_fit2,rmse_fit3,rmse_fit4,rmse_fit5);
par(mfrow = c(1,1));
plot(degree,rmse, xlab = "Degree of Polynomial", ylab = "RMSE", main = "RMSE using CV against Polynomial Degree");
lines(degree,rmse);

# RMSE against poly degree plot with fixed training and data set
degree = c(1,2,3,4,5);
rmse_f = c(rmse_fixed,rmse_fixed2,rmse_fixed3,rmse_fixed4,rmse_fixed5);
par(mfrow = c(1,1));
plot(degree,rmse_f, xlab = "Degree of Polynomial", ylab = "RMSE", main = "RMSE using fixed training and dataset against Polynomial Degree");
lines(degree,rmse_f);


#Lasso Regression
predictor_var = as.matrix(Boston[,1:13]); # Matrix of predictors
response_var = as.matrix(Boston[,14]); # Vector of response variable
lambda_seq = seq(from = 0.01, to = 0.1, by = 0.001); # range of lasso and ridge regression
medv_CVlasso_fit = cv.glmnet(x = predictor_var, y = response_var, alpha = 1, nfolds = 10, lambda = lambda_seq); # Lasso Regression with 10-fold CV
print(sqrt(min(medv_CVlasso_fit$cvm))); #Best RMSE

#Ridge Regression
medv_CVridge_fit = cv.glmnet(x = predictor_var, y = response_var, alpha = 0, nfolds = 10, lambda = lambda_seq); # Ridge Regression with 10-fold CV
print(sqrt(min(medv_CVridge_fit$cvm))); #Best RMSE








