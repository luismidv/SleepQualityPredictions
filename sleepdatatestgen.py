import pandas as pd
import numpy as np


#features_list = [User ID,Age,Gender,Sleep Quality,Bedtime,Wake-up Time,Daily Steps,Calories Burned,Physical Activity Level,Dietary Habits,Sleep Disorders,Medication Usage]

def datatest_generator_sleep():
    user_id = np.arange(101,201,1)
    print(user_id)
    age = np.random.randint(20,60,100)
    gender = [gender_list[i] for i in np.random.randint(0,2,100)]
    print(gender)

gender_list = ['f', 'm']
datatest_generator_sleep()