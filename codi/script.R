library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)

setwd("C:/Documents and Settings/rborras/Escritorio/1.ROGER/396.KTDMU/DADES")

# Create bag-of-apps in character string format, first by event,
# then merge to generate larger bags by device
cread  <- function(x) fread(x, colClasses = "character")
toStr  <- function(x) paste(x, collapse = ",")

app_ev <- cread("app_events.csv")
app_ev <- app_ev[ , .(apps = toStr(app_id)), by = event_id]

events <- cread("events.csv")
events <- merge(events, app_ev, by = "event_id", all.x = T)
events <- events[ , .(apps = toStr(apps)), by = device_id]

rm(app_ev)


# Merge bag-of-apps and brand data into train and test users 
users_train <- cread("gender_age_train.csv")
users_test  <- cread("gender_age_test.csv")
brands      <- cread("phone_brand_device_model.csv")
brands      <- brands[!duplicated(brands$device_id), ]

dmerge <- function(x, y) merge(x, y, by = "device_id", all.x = T)
users_train <- dmerge(users_train, events)
users_train <- dmerge(users_train, brands)
users_test  <- dmerge(users_test, events)
users_test  <- dmerge(users_test, brands)


# FeatureHash brand and app data to sparse matrix
b <- 2 ^ 14
f <- ~ phone_brand + device_model + split(apps, delim = ",") - 1
X_train <- hashed.model.matrix(f, users_train, b)
X_test  <- hashed.model.matrix(f, users_test,  b)

paste("Non-zero train features: ", sum(colSums(X_train) > 0))

# Validate xgboost model
Y_key <- sort(unique(users_train$group))
Y     <- match(users_train$group, Y_key) - 1

model <- sample(1:length(Y), 50000)
valid <- (1:length(Y))[-model]

param <- list(objective = "multi:softprob", num_class = 12,
              booster = "gblinear", eta = 0.01,
              eval_metric = "mlogloss")

dmodel <- xgb.DMatrix(X_train[model,], label = Y[model])
dvalid <- xgb.DMatrix(X_train[valid,], label = Y[valid])
watch  <- list(model = dmodel, valid = dvalid)

m1 <- xgb.train(data = dmodel, param, nrounds = 50,
                watchlist = watch)


# Use all train data and predict test
dtrain <- xgb.DMatrix(X_train, label = Y)
dtest  <- xgb.DMatrix(X_test)

m2 <- xgb.train(data = dtrain, param, nrounds = 50)

out <- matrix(predict(m2, dtest), ncol = 12, byrow = T)
out <- data.frame(device_id = users_test$device_id, out)
names(out)[2:13] <- Y_key
write.csv(out, file = "sub1.csv", row.names = F)
