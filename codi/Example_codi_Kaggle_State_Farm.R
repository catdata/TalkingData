library("doParallel")
library("foreach")
library("mxnet")
library("imager")
library("data.table")
library("nnet")
library("EBImage")
library("xgboost")
library("caret")

#directori

setwd("F:/KAGGLE_Distracted_Driver_Detection/IMATGES_EXEMPLE")

image_c0 <- readImage("c0_img_2093.jpg")
#display(image, method = "browser")
display(image_c0, method = "raster")

image_c1 <- readImage("c1_img_2477.jpg")
display(image_c1, method = "raster")

image_c2 <- readImage("c2_img_13859.jpg")
display(image_c2, method = "raster")

image_c3 <- readImage("c3_img_9853.jpg")
display(image_c3, method = "raster")

image_c5 <- readImage("c5_img_63.jpg")
display(image_c5, method = "raster")

image_c6 <- readImage("c6_img_2515.jpg")
display(image_c6, method = "raster")

image_c7 <- readImage("c7_img_629.jpg")
display(image_c7, method = "raster")

image_c8 <- readImage("c8_img_6884.jpg")
display(image_c8, method = "raster")

image_c9 <- readImage("c9_img_3512.jpg")
display(image_c9, method = "raster")

remove(list = ls())

setwd("D:/KAGGLE_Distracted_Driver_Detection/SINTAXI")

driver_details = read.csv("driver_imgs_list.csv");


mLogLoss.normalize = function(p, min_eta=1e-15, max_eta = 1.0){
  #min_eta
  for(ix in 1:dim(p)[2]) {
    p[,ix] = ifelse(p[,ix]<=min_eta,min_eta,p[,ix]);
    p[,ix] = ifelse(p[,ix]>=max_eta,max_eta,p[,ix]);
  }
  #normalize
  for(ix in 1:dim(p)[1]) {
    p[ix,] = p[ix,] / sum(p[ix,]);
  }
  return(p);
}

mlogloss = function(y, p, min_eta=1e-15,max_eta = 1.0){
  class_loss = c(dim(p)[2]);
  loss = 0;
  p = mLogLoss.normalize(p,min_eta, max_eta);
  for(ix in 1:dim(y)[2]) {
    p[,ix] = ifelse(p[,ix]>1,1,p[,ix]);
    class_loss[ix] = sum(y[,ix]*log(p[,ix]));
    loss = loss + class_loss[ix];
  }
  return (list("loss"=-1*loss/dim(p)[1],"class_loss"=class_loss));
}

mx.metric.mlogloss <- mx.metric.custom("mlogloss", function(label, pred){
  p = t(pred);
  m = mlogloss(class.ind(label),p);
  gc();
  return(m$loss);
})


img_height = 32;
img_width = 32;
use_rgb = FALSE;
num_channels = 1;

load.and.prep.img = function(tf){
  im = load.image(tf);
  if(!use_rgb){
    im = 0.2989 * channel(im,1) + 0.5870 * channel(im,2) + 0.1140 * channel(im,3) 
  }
  im = resize(im,img_width,img_height);
  return(im);
}

max_train_images_to_load = 1900;

cpu.cores = 6;
cl <- makeCluster(cpu.cores); 
registerDoParallel(cl);
train_img_matrix = 
  foreach(cls = 0:9, .packages=c('imager'), .combine=rbind, .multicombine=T) %dopar% {
    train_files = list.files(paste0("D:/KAGGLE_Distracted_Driver_Detection/train/c", cls, "/"),"*.*",full.names = T);
    train_files = train_files[1:max_train_images_to_load];
    targets = c();    
    m = data.frame(matrix(0,nrow=length(train_files),ncol=img_width*img_height*num_channels));
    mi = 1;
    for(tf in train_files){
      m[mi,] = as.numeric(load.and.prep.img(tf));
      targets[mi] = cls;
      mi = mi + 1;
    }             
    df = data.frame(m,stringsAsFactors = FALSE);
    df = cbind("target"=targets,df);
    df = cbind("file"=train_files,df);
    return(df);
  }
stopCluster(cl);

test_img_matrix = NULL;
test_id = NULL;
if(FALSE){
  cl <- makeCluster(cpu.cores); 
  registerDoParallel(cl);
  test_files = data.frame("file"=list.files("F:/KAGGLE_Distracted_Driver_Detection/test/","*.*",full.names = T),stringsAsFactors = FALSE);
  test_files$rid = 1:nrow(test_files);
  test_files$rid = test_files$rid%%7;
  test_img_matrix = 
    foreach(cls = 0:max(test_files$rid), .packages=c('imager'), .combine=rbind, .multicombine=T) %dopar% {
      t_files = test_files[test_files$rid==cls,"file"];
      m = data.frame(matrix(0,nrow=length(t_files),ncol=img_width*img_height*num_channels));
      mi = 1;
      for(tf in t_files){
        m[mi,] = as.numeric(load.and.prep.img(tf));
        mi = mi + 1;
      }             
      df = data.frame(m,stringsAsFactors = FALSE);
      df = cbind("file"=t_files,df);
      return(df);
    }
  stopCluster(cl);
  test_id = test_img_matrix[,1];
  test_img_matrix = test_img_matrix[,2:ncol(test_img_matrix)];
  
  test_img_matrix = t(test_img_matrix);
  dim(test_img_matrix) = c(img_width, img_height, num_channels, ncol(test_img_matrix));
}

driver_details = read.csv("driver_imgs_list.csv");
imgm = train_img_matrix;
imgm = imgm[order(sample(nrow(imgm))),];
y = imgm$target;
files = basename(as.character(imgm$file))
imgm = imgm[,3:ncol(imgm)];
mx = match(files, driver_details$img)
tgt = as.character(driver_details[mx,"classname"])

table(tgt==paste0("c",y));
subject = as.numeric(as.factor(driver_details[mx,"subject"]))

print(table(subject));

train_img_matrix$file <- NULL

#write.csv(train_img_matrix,  file = "train_img_matrix.csv")

sample_train <- sample(1:nrow(train_img_matrix), size=nrow(train_img_matrix)*0.75)

train_img_matrix_xg <- train_img_matrix[sample_train, ]
test_img_matrix_xg <- train_img_matrix[-sample_train, ]


xgb <- xgboost(data = data.matrix(train_img_matrix_xg[,-1]), 
               label = data.matrix(train_img_matrix_xg[,1]), 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softmax",
               num_class = 10,
               nthread = 6)

#Performance train

table(train_img_matrix_xg[,1])
y_pred_train <- predict(xgb, data.matrix(train_img_matrix_xg[,-1]))
table(y_pred_train, train_img_matrix_xg[,1])
confusionMatrix(table(y_pred_train, train_img_matrix_xg[,1]))

#Performance test

y_pred_test <- predict(xgb, data.matrix(test_img_matrix_xg[,-1]))
sum(table(y_pred_test, test_img_matrix_xg[,1]))
confusionMatrix(table(y_pred_test, test_img_matrix_xg[,1]))

#MXNET
#net configuration below
# input
data = mx.symbol.Variable('data')
conv1 = mx.symbol.Convolution(data=data, kernel=c(3,3), num_filter=20)
tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))

conv2 = mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                          kernel=c(2,2), stride=c(2,2))
flatten = mx.symbol.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
net = mx.symbol.SoftmaxOutput(data=fc2)


train_test_ratio = 0.8;
test_pred = NULL;
set.seed(1905);
res = c();
nfold = 3;
for(foldId in 1:nfold){
  smpl = sample(max(subject),floor(max(subject)*train_test_ratio))
  g_train = imgm[subject%in%smpl,];
  y_train = y[subject%in%smpl];
  g_test = imgm[!(subject%in%smpl),];
  y_test = y[!(subject%in%smpl)];
  
  g_train = t(g_train);
  g_test = t(g_test);
  
  dim(g_train) = c(img_width, img_height, num_channels, ncol(g_train));
  dim(g_test) = c(img_width, img_height, num_channels, ncol(g_test));
  
  start.time <- Sys.time();
  mx.set.seed(0);
  
  device = mx.cpu();
  model <- mx.model.FeedForward.create(
    X                  = g_train,
    y                  = y_train,
    eval.data          = list("data"=g_test,"label"=y_test),
    ctx                = device,
    symbol             = net,
    eval.metric        = mx.metric.mlogloss,
    num.round          = 3,
    learning.rate      = 0.05,
    momentum           = 0.01,
    wd                 = 0.0001,
    initializer=mx.init.uniform(0.1),
    array.batch.size   = 100,
    epoch.end.callback = mx.callback.save.checkpoint("statefarm"),
    batch.end.callback = mx.callback.log.train.metric(100),
    array.layout="columnmajor"
  );
  
  p = t(predict(model,g_test));
  m = mlogloss(class.ind(y_test),p);
  print(m);
  res[foldId] = m$loss;
  
  if(!is.null(test_img_matrix)){
    p_T = t(predict(model,test_img_matrix));
    if(is.null(test_pred)){
      test_pred = p_T;
    } else {
      test_pred = test_pred + p_T;
    }
  }
  
  end.time = Sys.time();
  time.taken <- end.time - start.time
  print(paste("net run time:",time.taken));
  model = NULL;
  gc();
}
print(paste("mean",mean(res),"sd",sd(res)));

if(!is.null(test_pred)){
  test_pred = test_pred /  foldId;
  #make a submission
  df = data.frame(test_pred); colnames(df) = c("c0","c1","c2","c3","c4","c5","c6","c7","c8","c9");
  df = cbind("img"=test_id,df);
  write.csv(df,"submission.csv",quote=F,row.names=F);
}    

setwd("C:/Documents and Settings/rborras/Escritorio/1.ROGER/386.KGL_CARS")

train_img_matrix <- read.csv("F:/KAGGLE_Distracted_Driver_Detection/train_img_matrix.csv")
train_img_matrix$X <- NULL

sample_train <- sample(1:nrow(train_img_matrix), size=nrow(train_img_matrix)*0.5)
sample_train <- sample(1:nrow(train_img_matrix), size=nrow(train_img_matrix)*0.2631579)

write.csv(train_img_matrix_xg, file = "Driver_Detection_sample.csv" )

train_img_matrix_xg <- train_img_matrix[sample_train, ]

test_img_matrix_xg <- train_img_matrix[-sample_train, ]


xgb_softmax <- xgboost(data = data.matrix(train_img_matrix_xg[,-1]), 
                       label = data.matrix(train_img_matrix_xg[,1]), 
                       eta = 0.1,
                       max_depth = 12, 
                       nround=10, 
                       subsample = 0.5,
                       colsample_bytree = 0.5,
                       seed = 1,
                       eval_metric = "merror",
                       objective = "multi:softmax",
                       num_class = 10,
                       nthread = 7)

#prediccio

y_pred_test <- predict(xgb_softmax, data.matrix(test_img_matrix_xg[,-1]))
sum(table(y_pred_test, test_img_matrix_xg[,1]))
confusionMatrix(table(y_pred_test, test_img_matrix_xg[,1]))
predict(xgb_softmax, train_img_matrix_xg[,-1])

xgb_softprob <- xgboost(data = data.matrix(train_img_matrix_xg[,-1]), 
                        label = data.matrix(train_img_matrix_xg[,1]), 
                        eta = 0.1,
                        max_depth = 12, 
                        nround=10, 
                        subsample = 0.5,
                        colsample_bytree = 0.5,
                        seed = 1,
                        eval_metric = "merror",
                        objective = "multi:softprob",
                        num_class = 10,
                        nthread = 1)


#Performance train

y_pred_train <- predict(xgb_softprob, data.matrix(train_img_matrix_xg[,-1]))
preds <- matrix(y_pred_train, ncol=10, byrow = TRUE)
max_pred <- apply(preds, 1, max)
summary(max_pred)

