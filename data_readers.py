
import pandas as pd



def MovieLens_reader(file_path):
    column_names = ['user id','movie id','rating','timestamp']
    df = pd.read_csv(file_path,sep='\t',header=None,names=column_names)
    return df

def MovieLens1M_reader(file_path):
    column_names = ['userId','movieId','rating','timestamp']
    df = pd.read_csv(file_path,sep=',',header=None,names=column_names)
    return df

def FilmTrust_reader(file_path):
    column_names = ['user id','movie id','rating','timestamp']
    df = pd.read_csv(file_path,sep=' ',header=None,names=column_names)
    return df


import pandas as pd
from datetime import datetime

def Netflix_reader(file_path):
    data = []
    current_user = None
    
    # خواندن فایل خط به خط
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # اگر خط با یک عدد و ":" شروع شود، این یک user_id جدید است
            if line.endswith(':'):
                current_user = int(line[:-1])  # حذف ":" و تبدیل به عدد
            else:
                # پارس کردن خط داده‌ها
                movie_id, rating, date_str = line.split(',')
                movie_id = int(movie_id)
                rating = int(rating)
                # تبدیل تاریخ به Unix timestamp
                date = datetime.strptime(date_str, '%Y-%m-%d')
                timestamp = int(date.timestamp())
                # افزودن به لیست داده‌ها
                data.append([current_user, movie_id, rating, timestamp])
    
    # ایجاد DataFrame
    column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.DataFrame(data, columns=column_names)
    
    # شافل کردن ردیف‌های DataFrame
    df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    
    return df








# def Netflix_reader(file_path):
#     data = []
#     current_user = None
    
#     # خواندن فایل خط به خط
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             # اگر خط با یک عدد و ":" شروع شود، این یک user_id جدید است
#             if line.endswith(':'):
#                 current_user = int(line[:-1])  # حذف ":" و تبدیل به عدد
#             else:
#                 # پارس کردن خط داده‌ها
#                 movie_id, rating, date_str = line.split(',')
#                 movie_id = int(movie_id)
#                 rating = int(rating)
#                 # تبدیل تاریخ به Unix timestamp
#                 date = datetime.strptime(date_str, '%Y-%m-%d')
#                 timestamp = int(date.timestamp())
#                 # افزودن به لیست داده‌ها
#                 data.append([current_user, movie_id, rating, timestamp])
    
#     # ایجاد DataFrame به جای آرایه NumPy
#     column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
#     df = pd.DataFrame(data, columns=column_names)
#     return df





# import pandas as pd
# import numpy as np
# from datetime import datetime

# def Netflix_reader(file_path):
#     data = []
#     current_user = None
    
#     # خواندن فایل خط به خط
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             # اگر خط با یک عدد و ":" شروع شود، این یک user_id جدید است
#             if line.endswith(':'):
#                 current_user = int(line[:-1])  # حذف ":" و تبدیل به عدد
#             else:
#                 # پارس کردن خط داده‌ها
#                 movie_id, rating, date_str = line.split(',')
#                 movie_id = int(movie_id)
#                 rating = int(rating)
#                 # تبدیل تاریخ به Unix timestamp
#                 date = datetime.strptime(date_str, '%Y-%m-%d')
#                 timestamp = int(date.timestamp())
#                 # افزودن به لیست داده‌ها
#                 data.append([current_user, movie_id, rating, timestamp])
    
#     # تبدیل لیست به آرایه NumPy
#     return data

# # مثال استفاده:
# # فرض کنید فایل شما به نام 'data.txt' ذخیره شده است
# # df = Netflix_reader('data.txt')
# # print(df)