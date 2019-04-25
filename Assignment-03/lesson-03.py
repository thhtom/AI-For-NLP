
import  pandas as pd
import  matplotlib.pyplot as plt
import  numpy as np
import  random


content = pd.read_csv("/Users/tom/Desktop/data/train.scv")
content = content.dropna()
age_with_fares = content[(content["Age"]>22)&(content["Fare"]>400)&(content["Fare"]<120)]
print(age_with_fares)

sub_age = age_with_fares["Age"]
sub_Fare = age_with_fares["Fare"]

def func(age,k,b): return  k*age + b

def loss(y,yhat):

    return  np.mean(np.abs(y-yhat))

min_error_rate = float("inf")

loop_times = 10000

losses =  []

change_direction =[
    (+1,-1),
    (+1,+1),
    (-1,+1),
    (-1,-1)
]

k_hat = random.random()*20 - 10
b_hat = random.random()*20 - 10

best_k,best_b = k_hat, b_hat

best_direction = None

def step(): return  random.random()*1

direction = random.choice(change_direction)

def derivate_k(y,yhat,k):
    abs_values = [1 if (y_i - yhat_i)>0 else -1  for y_i,yhat_i in zip(y,yhat)]
    return  np.mean([a* -x_i for a,x_i in zip(abs_values,x)])

def derivate_b(y,yhat):
    abs_values = [1 if (y_i - yhat_i)>0 else -1 for y_i,yhat_i in zip(y,yhat)]
    return  np.mean([a*-1 for a in abs_values])

learn_rate = 1e-1

while loop_times > 0:
    k_delta = -1*learn_rate*derivate_k(sub_Fare,func(sub_Fare,k_hat,b_hat),sub_age)
    b_delta = -1*learn_rate*derivate_b(sub_Fare,func(sub_age,k_hat,b_hat))

    k_hat += k_delta
    b_hat += b_delta

estimated_fares = func(sub_age,k_hat,b_hat)
error_rate = loss(y = sub_Fare,yhat = estimated_fares)

print('loop == {}'.format(loop_times))
print('f(age) = {}*age + {},with error rate :{}'.format(best_k,best_b,error_rate))

losses.append(error_rate)

loop_times -= 1

plt.plot(range(len(losses)),losses)
plt.show()



