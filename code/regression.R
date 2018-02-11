library(ggplot2)
df = read.table('./data/ex1data1.txt',sep = ',',header = F)
names(df) = c('population', 'profit')
head(df)
sp = ggplot(df,aes(x = population,y = profit))+geom_point()
#+geom_point()+stat_smooth(method = lm,se = T)


normalize_feature = function(df){
  return(scale(df))
}

get_x  = function(df){

  df = data.frame(df)
  ones = rep(1,nrow(df))
  df = cbind(ones,df)
  col_num = ncol(df)
  df = df[,-col_num]
  return( as.matrix(df,nrow = nrow(df),ncol = ncol(df)) )
  
}


get_y = function(df){
  col_num = ncol(df)
  return(as.matrix(df[,col_num],ncol = 1))
}


X =  get_x(df)
Y = get_y(df)
theta = as.matrix(c(0,0),ncol = 1 )

lr_cost = function(theta, x, y){
  m = nrow(x)
  inner = x%*%theta - y  # R(m*1)，X @ theta等价于X.dot(theta)
  # 1*m @ m*1 = 1*1 in matrix multiplication
  # but you know numpy didn't do transpose in 1d array, so here is just a
  # vector inner product to itselves
  square_sum = t(inner)%*% inner
  cost = square_sum / (2 * m)
  return(cost)
  
} 


lr_cost(theta,X,Y)
gradient = function(theta, x, y){
  m = nrow(x) #
  inner = t(x)%*%(x%*%theta -y)
  return(inner/m)
}

gradient(theta,X,Y)
#===========================
#batch gradient decent
#===========================
batch_gradient_decent = function(theta,x,y,alpah = 0.01,epoch){
  
  cost_data = lr_cost(theta,x,y)
  theta_copy = theta
  for(i in 1:epoch){
    theta_copy = theta_copy - alpah*gradient(theta_copy,x,y)
    cost_mid = lr_cost(theta_copy,x,y)
    cost_data = rbind(cost_data,cost_mid)
  }
  return(list(theta_copy,cost_data))
}

final_data = batch_gradient_decent(theta,X,Y,alpah = 0.01,epoch = 500)
final_theta = final_data[[1]]
final_cost = final_data[[2]]
b =  final_theta[1]
w = final_theta[2] 
lm_pred = X%*%final_theta

ts.plot(ts(final_cost[1:10]))
sp + geom_line(data = data.frame(population = df$population,profit = lm_pred),colour = 'red',size = .8)+
  stat_smooth(method = lm,se = F) 

#==========================
#Stochastic gradient decent
#==========================
stoc_gradient_decent = function(theta,x,y,alpah = 0.01){
  
  cost_data = lr_cost(theta,x[1,,drop = FALSE],y[1,,drop = FALSE])
  theta_copy = theta
  m = nrow(x)
  for(i in 1:m){
    theta_copy = theta_copy - alpah*gradient(theta_copy,x[i,,drop = FALSE],y[i,,drop = FALSE])
    cost_mid = lr_cost(theta_copy,x[i,,drop = FALSE],y[i,,drop = FALSE])
    cost_data = rbind(cost_data,cost_mid)
  }
  return(list(theta_copy,cost_data))
}
stoc_gradient_decent(theta,X,Y,alpah = 0.01)

stoc_result = stoc_gradient_decent(theta,X,Y,alpah = 0.01)
stoc_theta = stoc_result[[1]]
stoc_cost = stoc_result[[2]]
ts.plot(ts(stoc_cost))
stoc_pred =  X%*%final_theta
sp + geom_line(data = data.frame(population = df$population,profit = stoc_pred),colour = 'yellow',size = .8)+
  stat_smooth(method = lm,se = F) 

#============================ 
#Mini-Batch Gradient Descent
#============================
minibatch_gradient_decent = function(theta,x,y,alpah = 0.01,epoch){
  j = sample(3,nrow(x),replace = T)
  x = x[j==1,,drop = FALSE]
  y = y[j==1,,drop = FALSE]
  cost_data = lr_cost(theta,x,y)
  theta_copy = theta
  for(i in 1:epoch){
    theta_copy = theta_copy - alpah*gradient(theta_copy,x,y)
    cost_mid = lr_cost(theta_copy,x,y)
    cost_data = rbind(cost_data,cost_mid)
  }
  return(list(theta_copy,cost_data))
}


minibatch_result = minibatch_gradient_decent(theta,X,Y,alpah = 0.01,epoch = 500)
minibatch_theta = minibatch_result[[1]]
minibatch_cost = minibatch_result[[2]]
ts.plot(ts(minibatch_cost))

#==========================
#normal equation
#==========================
summary(lm(profit~population,data = df))
