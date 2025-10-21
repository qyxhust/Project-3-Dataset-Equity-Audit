import pandas as pd
import math


df = pd.read_csv("population_summary.csv")

shannon_sub = -sum(row * math.log(row) for row in df["percent"])
shannon_super = -sum(row * math.log(row) for row in df["super_percent"])

print(f" Shannon sub: {shannon_sub}, Shannon super: {shannon_super}")


"""
Notes: 

Shannon sub: 3.2765669523856547, Shannon super: 8.363839112542903

Both Shannon indicies for the sub and super populations are pretty high, indicating that the dataset is pretty diverse geographically-wise

"""