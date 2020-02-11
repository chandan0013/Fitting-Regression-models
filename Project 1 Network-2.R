#import data file (change path to correct directory)
network_backup_dataset<-read.csv("/home/usha/network_backup_dataset.csv", header=TRUE)

# Relationship plot, Piecewise Linear,Polynomial Fit
# for Network Data Set
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

# Relationship plot

# Splitting the data set according to Workflow

Work= split(network_backup_dataset, f = network_backup_dataset$Work.Flow.ID, drop = FALSE);
WorkFlow_0 = Work$work_flow_0;
WorkFlow_1 = Work$work_flow_1;
WorkFlow_2 = Work$work_flow_2;
WorkFlow_3 = Work$work_flow_3;
WorkFlow_4 = Work$work_flow_4;

# Splitting the data set according to Files

WorkFlow_0_File = split(WorkFlow_0, f = WorkFlow_0$File.Name, drop = FALSE);
WorkFlow_1_File = split(WorkFlow_1, f = WorkFlow_1$File.Name, drop = FALSE);
WorkFlow_2_File = split(WorkFlow_2, f = WorkFlow_2$File.Name, drop = FALSE);
WorkFlow_3_File = split(WorkFlow_3, f = WorkFlow_3$File.Name, drop = FALSE);
WorkFlow_4_File = split(WorkFlow_4, f = WorkFlow_4$File.Name, drop = FALSE);

# Selecting a file from each Work Flow
WorkFlow_0_File_1 = WorkFlow_0_File$File_1;
WorkFlow_1_File_8 = WorkFlow_1_File$File_8;
WorkFlow_2_File_13 = WorkFlow_2_File$File_13;
WorkFlow_3_File_20 = WorkFlow_3_File$File_20;
WorkFlow_4_File_26 = WorkFlow_4_File$File_26;

# Model training to be used in poly regression without CV

smp_size_network <- floor(0.75 * nrow(network_backup_dataset));
set.seed(123);
train_ind_network <- sample(seq_len(nrow(network_backup_dataset)), size = smp_size_network);

train_network <- network_backup_dataset[train_ind_network, ];
test_network <- network_backup_dataset[-train_ind_network, ];

# Formula

network_form_piece = formula(Size.of.Backup..GB. ~ Week.. + Day.of.Week + Backup.Start.Time...Hour.of.Day + File.Name +Backup.Time..hour.); # Formula for piecewise regression
network_form = formula(Size.of.Backup..GB. ~ Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.); # Formula for linear regression

# Linear Regression with CV
size_fit_all = lm(network_form, data = network_backup_dataset, contrasts = "treatment");
size_CVlm_fit_all = CVlm(data = network_backup_dataset, form.lm = size_fit_all, m = 10, printit = FALSE);
rmse_lin_network = rmse(size_CVlm_fit_all$Size.of.Backup..GB., size_CVlm_fit_all$cvpred);# RMSE
res_fit_network_all = size_CVlm_fit_all$Size.of.Backup..GB. - size_CVlm_fit_all$cvpred;
par(mfrow = c(2,1));
plot(size_CVlm_fit_all$Size.of.Backup..GB., size_CVlm_fit_all$cvpred, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Network)");
abline(0,1, untf = FALSE);
plot(size_CVlm_fit_all$cvpred, res_fit_network_all, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Network)");
abline(h = 0, untf = FALSE);

# Linear Regression with fixed training and dataset
size_fit_fixed = lm(network_form,data = train_network);
size_predict_fixed = predict.lm(size_fit_fixed,test_network);
rmse_fixed_network = rmse(test_network$Size.of.Backup..GB., size_predict_fixed);


# Quadratic Regression with CV
network_form2 = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)+((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2));
size_fit_poly2 = lm(network_form2,data = network_backup_dataset); 
size_CVlm_poly2 = CVlm(data = network_backup_dataset, form.lm = size_fit_poly2, m=10,printit = FALSE);
rmse_network_poly2 = rmse(size_CVlm_poly2$Size.of.Backup..GB., size_CVlm_poly2$cvpred, na.rm = TRUE);

# Quadratic Regression with fixed training and dataset
network_form2_fixed = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)+((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2));
size_fit_fixed2 = lm(network_form2_fixed,data = train_network);
size_predict_fixed2 = predict.lm(size_fit_fixed2,test_network);
rmse_fixed2_network = rmse(test_network$Size.of.Backup..GB., size_predict_fixed2, na.rm = TRUE);

# Cubic Regression with CV
network_form3 = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^3)); 
size_fit_poly3 = lm(network_form3,data = network_backup_dataset);
size_CVlm_poly3 = CVlm(data = network_backup_dataset, form.lm = size_fit_poly3, m=10,printit = FALSE);
rmse_network_poly3 = rmse(size_CVlm_poly3$Size.of.Backup..GB., size_CVlm_poly3$cvpred, na.rm = TRUE);

#Cubic Regression with fixed training and dataset
network_form3_fixed = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^3));
size_fit_fixed3 = lm(network_form3_fixed,data = train_network);
size_predict_fixed3 = predict.lm(size_fit_fixed3,test_network);
rmse_fixed3_network = rmse(test_network$Size.of.Backup..GB., size_predict_fixed3, na.rm = TRUE);

#Quartic Regression with CV
network_form4 = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^3) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.)^4)); 
size_fit_poly4 = lm(network_form4,data = network_backup_dataset);
size_CVlm_poly4 = CVlm(data = network_backup_dataset, form.lm = size_fit_poly4, m=10,printit = FALSE);
rmse_network_poly4 = rmse(size_CVlm_poly4$Size.of.Backup..GB., size_CVlm_poly4$cvpred, na.rm = TRUE);

#Quartic Regression with fixed training and dataset
network_form4_fixed = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^3) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.)^4));
size_fit_fixed4 = lm(network_form4_fixed,data = train_network);
size_predict_fixed4 = predict.lm(size_fit_fixed4,test_network);
rmse_fixed4_network = rmse(test_network$Size.of.Backup..GB., size_predict_fixed4, na.rm = TRUE);

#Penta Regression with CV
network_form5 = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^3) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.)^4) +((Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.)^5));
size_fit_poly5 = lm(network_form5,data = network_backup_dataset);
size_CVlm_poly5 = CVlm(data = network_backup_dataset, form.lm = size_fit_poly5, m=10,printit = FALSE);
rmse_network_poly5 = rmse(size_CVlm_poly5$Size.of.Backup..GB., size_CVlm_poly5$cvpred, na.rm = TRUE);

#Penta Regression with fixed training and dataset
network_form5_fixed = formula(Size.of.Backup..GB. ~ (Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^2) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day + Work.Flow.ID +Backup.Time..hour.)^3) + ((Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.)^4) +((Day.of.Week + Backup.Start.Time...Hour.of.Day +Work.Flow.ID + Backup.Time..hour.)^5));
size_fit_fixed5 = lm(network_form5_fixed,data = train_network);
size_predict_fixed5 = predict.lm(size_fit_fixed5,test_network);
rmse_fixed5_network = rmse(test_network$Size.of.Backup..GB., size_predict_fixed5, na.rm = TRUE);

#Plotting RMSE(10-fold) against poly degree
rmse_poly = c(rmse_lin_network,rmse_network_poly2,rmse_network_poly3,rmse_network_poly4,rmse_network_poly5);
poly_degree = c(1,2,3,4,5);
par(mfrow = c(1,1));
plot(poly_degree,rmse_poly, xlab = "Polynomial Degree", ylab = "RMSE", main = "RMSE(10-fold) vs Degree");
lines(poly_degree,rmse_poly);


# Piecewise Linear Fit


#Workflow 0
size_CVlm_fit_0 = CVlm(data = WorkFlow_0, form.lm = network_form_piece, m = 10, printit = FALSE);
res_wf_0 = rmspe(size_CVlm_fit_0$Size.of.Backup..GB., size_CVlm_fit_0$cvpred); #RMSE
res_fit_network_work_0 = size_CVlm_fit_0$Size.of.Backup..GB. - size_CVlm_fit_0$cvpred;
par(mfrow = c(2,1));
plot(size_CVlm_fit_0$Size.of.Backup..GB., size_CVlm_fit_0$cvpred, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Workflow 0)");
abline(0,1, untf = FALSE);
plot(size_CVlm_fit_0$cvpred, res_fit_network_work_0, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Workflow 0)");
abline(h = 0, untf = FALSE);



#Workflow 1
size_CVlm_fit_1 = CVlm(data = WorkFlow_1, form.lm = network_form_piece, m = 10, printit = FALSE);
res_wf_1 = rmspe(size_CVlm_fit_1$Size.of.Backup..GB., size_CVlm_fit_1$cvpred); #RMSE
res_fit_network_work_1 = size_CVlm_fit_1$Size.of.Backup..GB. - size_CVlm_fit_1$cvpred;
par(mfrow = c(2,1));
plot(size_CVlm_fit_1$Size.of.Backup..GB., size_CVlm_fit_1$cvpred, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Workflow 1)");
abline(0,1, untf = FALSE);
plot(size_CVlm_fit_1$cvpred, res_fit_network_work_1, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Workflow 1)");
abline(h = 0, untf = FALSE);

#Workflow 2
size_CVlm_fit_2 = CVlm(data = WorkFlow_2, form.lm = network_form_piece, m = 10, printit = FALSE);
res_wf_2 = rmspe(size_CVlm_fit_2$Size.of.Backup..GB., size_CVlm_fit_2$cvpred); #RMSE
res_fit_network_work_2 = size_CVlm_fit_2$Size.of.Backup..GB. - size_CVlm_fit_2$cvpred;
par(mfrow = c(2,1));
plot(size_CVlm_fit_2$Size.of.Backup..GB., size_CVlm_fit_2$cvpred, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Workflow 2)");
abline(0,1, untf = FALSE);
plot(size_CVlm_fit_2$cvpred, res_fit_network_work_2, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Workflow 2)");
abline(h = 0, untf = FALSE);

#Workflow 3
size_CVlm_fit_3 = CVlm(data = WorkFlow_3, form.lm = network_form_piece, m = 10, printit = FALSE);
res_wf_3 = rmspe(size_CVlm_fit_3$Size.of.Backup..GB., size_CVlm_fit_3$cvpred); #RMSE
res_fit_network_work_3 = size_CVlm_fit_3$Size.of.Backup..GB. - size_CVlm_fit_3$cvpred;
par(mfrow = c(2,1));
plot(size_CVlm_fit_3$Size.of.Backup..GB., size_CVlm_fit_3$cvpred, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Workflow 3)");
abline(0,1, untf = FALSE);
plot(size_CVlm_fit_3$cvpred, res_fit_network_work_3, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Workflow 3)");
abline(h = 0, untf = FALSE);

#Workflow 4
size_CVlm_fit_4 = CVlm(data = WorkFlow_4, form.lm = network_form_piece, m = 10, printit = FALSE);
res_wf_4 = rmspe(size_CVlm_fit_4$Size.of.Backup..GB., size_CVlm_fit_4$cvpred); #RMSE
res_fit_network_work_4 = size_CVlm_fit_4$Size.of.Backup..GB. - size_CVlm_fit_4$cvpred;
par(mfrow = c(2,1));
plot(size_CVlm_fit_4$Size.of.Backup..GB., size_CVlm_fit_4$cvpred, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Workflow 4)");
abline(0,1, untf = FALSE);
plot(size_CVlm_fit_4$cvpred, res_fit_network_work_4, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Workflow 4)");
abline(h = 0, untf = FALSE);


# Random Forest

size_random_fit = randomForest(formula = network_form, data = network_backup_dataset, ntree = 35, mtry = 4, importance = TRUE);
random_prediction = size_random_fit$predicted;
random_residual = network_backup_dataset$Size.of.Backup..GB. - random_prediction;
par(mfrow = c(2,1));
plot(network_backup_dataset$Size.of.Backup..GB.,random_prediction, xlab = "Actual", ylab = "Fitted", main = "Fitted vs Actual (Network - Random Forest)");
abline(0,1, untf = FALSE);
plot(random_prediction, random_residual, xlab = "Fitted", ylab = "Residual", main = "Fitted vs Residual (Network - Random Forest)");
abline(h = 0, untf = FALSE); 
NROW(random_prediction);
sqrt(min(size_random_fit$mse)); #RMSE



#Neural Network (Takes a while to run)

#import data file (change path to correct directory)
data1<-read.csv("/home/usha/network_backup_dataset.csv", header=TRUE)
require(quantmod) #for Lag()
require(nnet)
require(caret)
#fit control using 10 fold cross validation
fitControl <- trainControl(method = "cv", classProbs = TRUE, number = 10)

#Fit model
#find ideal number of nodes,decay and train
#nnGrid<- expand.grid(.size = c(10,15,20,21),.decay = c(0,0.0001,0.001,0.01))
#model <- train(data1$Size.of.Backup..GB.~., data = data1, method='nnet',preProcess = "center",trainControl = fitControl, linout=TRUE, trace = FALSE,tuneGrid=nnGrid)

#once the ideal combination of hyperparameters is found we can simply train on those parameters:
nnGridOptimal<-expand.grid(.size = 20,.decay = 0.01)
model <- train(data1$Size.of.Backup..GB.~., data = data1, method='nnet',preProcess = "center",trainControl = fitControl, linout=TRUE, trace = FALSE,tuneGrid=nnGridOptimal)

fitted_data <- predict(model, data1)
#plot fitted data against actual data
plot(data1$Size.of.Backup..GB.,fitted_data,ylab = "Fitted Data",xlab = "Actual Data",main = "Fitted vs Actual Data for Neural Network");
abline(0,1,untf=FALSE);

#Examine results
model
