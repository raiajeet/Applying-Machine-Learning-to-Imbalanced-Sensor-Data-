rm(list = ls())

#datasets
terastub1=read.csv("C:\\Users\\AJIT\\Documents\\EDA\\teras-TURB-7_20171129-121430.csv",header=TRUE,stringsAsFactors =FALSE, sep=",") 
terastub2=read.csv("C:\\Users\\AJIT\\Documents\\EDA\\teras-TURB-8_20171129-122236.csv",header=TRUE,stringsAsFactors =FALSE,sep=",") 
terastub3=read.csv("C:\\Users\\AJIT\\Documents\\EDA\\teras-turb-9_20171129-123051.csv",header=TRUE,stringsAsFactors =FALSE, sep=",") 

#Avoiding unneccessary features
terastub1=subset(terastub1,select=-c(X,GEN.EE.BRG.TEMP))
terastub2=subset(terastub2,select=-c(X,Date.Time,TUR.ROTATIONL.VELOCITY.2850RPM,
                                       TUR.ROTATIONL.VELOCITY.1200RPM))
terastub3=subset(terastub3,select=-c(X,Date.Time,IDF.B.IN.MOT.CTRL.DMP.CTRL.CMD,IDF.A.HYD.CPL.CTRL.CMD))
TUR=cbind(terastub1,terastub2,terastub3)

#Actual data
data=as.data.frame(TUR)
write.csv(data, file = "C:\\Users\\AJIT\\Documents\\rpdata.csv", row.names = FALSE)
sum(is.na(data))
data=na.omit(data)

# Data Pre-processing
data$Date.Time <- format(as.Date(data$Date.Time),"%d%b%Y %H:%M:%S")
data=data[order(as.Date(data$Date.Time, format="%d%b%Y %H:%M:%S")),]
data$Date.Time=as.Date(data$Date.Time,"%d%b%Y %H:%M:%S")
#data=aggregate(.~Date.Time, data=data, mean)
data$class<-0
data$class[data$Date.Time=="2017-07-15"]<-1
data$class[data$Date.Time=="2017-02-14"]<-1
data$class[data$Date.Time=="2017-11-09"]<-1
data$class[data$Date.Time=="2017-10-11"]<-1
table(data$class)
data=subset(data,select = -c(Date.Time))
data=data[ , apply(data, 2, var) != 0]

#getting train and test data
data$class=as.factor(data$class)
round(prop.table(table(data$class)) * 100, digits = 1)
id=sample(2,nrow(data),prob=c(.8,.3),replace=TRUE)
train=data[id==1,]
test=data[id==2,]

#Principal Component Analysis
pc<- prcomp(train[-28],center=TRUE, scale. = TRUE)
summary(pc)
sd=pc$sdev
var=sd*sd
prop=var/sum(var)
plot(cumsum(prop), xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b",col="blue")
library(devtools)
library(ggbiplot)
g=ggbiplot(pc, obs.scale = 1, var.scale = 1, 
           ellipse = TRUE, 
           circle = TRUE,ellipse.prob = 0.68,groups =train$class,col="green")
g=g+scale_color_discrete(name="")
g
trg=predict(pc,train)
trg=trg[,1:14]
train=data.frame(trg,train[28])
tst=predict(pc,test)
tst=tst[,1:14]
test=data.frame(tst,test[28])

#Models

#Decision Tree
library(ROSE)
library(rpart)
treeimb <- rpart(class ~ ., data = train)
pred.treeimb <- predict(treeimb, newdata =test,type = "prob")
accuracy.meas(test$class,pred.treeimb[,2])
roc.curve(test$class, pred.treeimb[,2], plotit = T)

#over sampling
data_balanced_over <- ovun.sample(class ~ ., data = train, method = "over",N = 50374)$data
table(data_balanced_over$class)

#Under Sampling
data_balanced_under <- ovun.sample(class ~ ., data =train, method = "under", N = 418, seed = 1)$data
table(data_balanced_under$class)
data.rose <- ROSE(class ~ ., data = train, seed = 1)$data
table(data.rose$class)

tree.rose <- rpart(class ~ ., data = data.rose)
tree.over <- rpart(class ~ ., data = data_balanced_over)
tree.under <- rpart(class ~ ., data = data_balanced_under)

#make predictions on unseen data
pred.tree.rose <- predict(tree.rose, newdata = test)
pred.tree.over <- predict(tree.over, newdata = test)
pred.tree.under <- predict(tree.under, newdata = test)
accuracy.meas(test$class,pred.tree.under[,2])
accuracy.meas(test$class,pred.tree.over[,2])
accuracy.meas(test$class,pred.tree.rose[,2])


roc.curve(test$class, pred.tree.under[,2])
roc.curve(test$class, pred.tree.rose[,2])
roc.curve(test$class, pred.tree.over[,2],col="blue")

#Logistic Regression
glm <- glm(class ~ ., data = train,family = binomial)
summary(glm)
predict <- predict(glm,test, type = 'response')
accuracy.meas(test$class,predict)
roc.curve(test$class,predict,col="green")

glm.rose <- glm(class ~ ., data = data.rose,family = binomial)
glm.over <- glm(class ~ ., data = data_balanced_over,family = binomial)
glm.under <- glm(class ~ ., data = data_balanced_under,family = binomial)

#make predictions on unseen data
pred.glm.rose <- predict(glm.rose, newdata = test,type="response")
pred.glm.over <- predict(glm.over, newdata = test,type="response")
pred.glm.under <- predict(glm.under, newdata = test,type="response")

accuracy.meas(test$class,pred.glm.rose)
accuracy.meas(test$class,pred.glm.over)
accuracy.meas(test$class,pred.glm.under)

roc.curve(test$class,pred.glm.rose,col="green")
roc.curve(test$class, pred.glm.over,col="red")
roc.curve(test$class, pred.glm.under,col="blue")

#Random Forest
library(randomForest)
rf <- randomForest(class ~ ., data = train)
pred.rf <- predict(rf, newdata =test,type="prob")
accuracy.meas(test$class,pred.rf[,2])
roc.curve(test$class, pred.rf[,2], plotit = T)


library(randomForest)
rf.rose <- randomForest(class ~ ., data = data.rose)
rf.over <- randomForest(class ~ ., data = data_balanced_over)
rf.under <- randomForest(class ~ ., data = data_balanced_under)

#make predictions on unseen data
pred.rf.rose <- predict(rf.rose, newdata = test,type="prob")
pred.rf.over <- predict(rf.over, newdata = test,type="prob")
pred.rf.under <- predict(rf.under, newdata = test,type="prob")

accuracy.meas(test$class,pred.rf.rose[,2])
accuracy.meas(test$class,pred.rf.over[,2])
accuracy.meas(test$class,pred.rf.under[,2])

roc.curve(test$class,pred.rf.rose[,2],col="green")
roc.curve(test$class, pred.rf.over[,2],col="red")
roc.curve(test$class, pred.rf.under[,2],col="black")

