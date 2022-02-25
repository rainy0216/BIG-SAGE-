import math

from deprecated.sphinx import deprecated
from stellargraph import StellarGraph
import pandas as pd
import numpy as np
import pickle
import os
import time, datetime
import asyncio
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

class TimeAwaredDataset:
    # 选取源数据集的前N行生成可训练数据的时候，指定的num_edges可选用预定义的四种大小
    SIZE_MINI_MIN = 10000
    SIZE_MINI_LIT = 100000
    SIZE_MINI_MOR = 500000
    SIZE_MINI_MAX = 1000000

    # ↓清理指定目录下的缓存文件，del_result=True时包括删除TAD/TCAD文件(请谨慎操作)
    @staticmethod
    def cache_clear(data_name, del_result=False):
        print("Start to clean cache/temp files...")

        source_files = {'amazon-books': ['ratings.csv', 'reviews.json'], 'amazon-cds': ['ratings.csv', 'reviews.json'],
                        'amazon-foods': ['ratings.csv', 'reviews.json'],
                        'amazon-movies': ['ratings.csv', 'reviews.json'],
                        'amazon-musics': ['ratings.csv', 'reviews.json'],
                        'amazon-toys': ['ratings.csv', 'reviews.json']}
        result_files = {'amazon-books': ['BOOKS.TAD', 'BOOKS.TCAD', 'BOOKS_MINI.TAD', 'BOOKS_MINI.TCAD'],
                        'amazon-cds': ['CDS.TAD', 'CDS.TCAD', 'CDS_MINI.TAD', 'CDS_MINI.TCAD'],
                        'amazon-foods': ['FOODS.TAD', 'FOODS.TCAD', 'FOODS_MINI.TAD', 'FOODS_MINI.TCAD'],
                        'amazon-movies': ['MOVIES.TAD', 'MOVIES.TCAD', 'MOVIES_MINI.TAD', 'MOVIES_MINI.TCAD'],
                        'amazon-musics': ['MUSICS.TAD', 'MUSICS.TCAD', 'MUSICS_MINI.TAD', 'MUSICS_MINI.TCAD'],
                        'amazon-toys': ['TOYS.TAD', 'TOYS.TCAD', 'TOYS_MINI.TAD', 'TOYS_MINI.TCAD']}
        path = 'data/'
        if data_name not in source_files.keys() or data_name not in result_files.keys():
            print("ERROR: Please make sure the data_name is correct!")
        safe_files = source_files[data_name]
        if del_result is not True:
            safe_files += result_files[data_name]
        num = 0
        for roots, dirs, files in os.walk(path + data_name):
            for file in files:
                if file not in safe_files:
                    file_path = path + data_name + '/' + file
                    if os.access(file_path, os.F_OK) is True:
                        os.remove(file_path)
                        num += 1
        print("Cleaning has removed " + str(num) + " files")

    # ↓直接指定的目标数据集(仅支持拓展名为TAD/TCAD的对象持久化文件)
    @staticmethod
    def load(path):
        type = path[path.rfind('.') + 1:]
        if type not in ['TAD', 'TCAD']:
            print()
            exit(0)
        if os.access(path, os.F_OK):
            print('Serialized data object detected: \n"' + path + '"')
            with open(file=path, mode='rb') as fr:
                return pickle.load(file=fr)

    # ↓从现有数据生成可训练数据，保存为TAD文件并返回可训练数据
    '''
    # 生成拓展名为TAD的对象持久化文件，需要提供用户、项目特征矩阵、交互矩阵(含分数+不含分数)和时间戳矩阵(请自行做好归一化处理)
    # 所有矩阵均以panda data frame的格式传入，其中要求如下
    # users: index即user的id，所有数据列均为特征数据
    # items: index即item的id，所有数据列均为特征数据
    # ratings: 包含三列：user id，item id，评分(无需归一化)
    # edges: 包含三列：user id，item id，连接(1/0)
    # timestamps: 包含三列：user id，item id，归一化时间戳
    # 可训练数据包括G_rating, G_timestamp, ratings，两个StellarGraph对象(不包含标签的图结构对象)和一个pandas data frame(包含标签的数据)
    '''

    @staticmethod
    def generator_TAD_from_files(users, items, ratings, edges, timestamps, path="OUTPUT.TAD"):
        # 创建StellarGraph对象
        # print(df_users)
        G_rating = StellarGraph(nodes={'source': users, 'target': items},
                                edges={'weight': edges})
        G_timestamp = StellarGraph(nodes={'source': users, 'target': items},
                                   edges={'weight': timestamps})
        # 数据检查
        G_rating.check_graph_for_ml()
        G_timestamp.check_graph_for_ml()
        # 数据对象序列化存档
        with open(file=path, mode='wb') as fw:
            pickle.dump(obj=(G_rating, G_timestamp, ratings), file=fw)
        print('Data has been preprocessed successfully.')
        return G_rating, G_timestamp, ratings

    # ↓从现有数据生成可训练数据，保存为TCAD文件并返回可训练数据
    '''
    # 生成拓展名为TCAD的对象持久化文件，需要提供用户、项目特征矩阵、交互矩阵(含分数+不含分数)和时间戳矩阵(请自行做好归一化处理)
    # 所有矩阵均以panda data frame的格式传入，其中要求如下
    # users: index即user的id，所有数据列均为特征数据
    # items: index即item的id，所有数据列均为特征数据
    # ratings: 包含三列：user id，item id，评分(无需归一化)
    # edges: 包含三列：user id，item id，连接(1/0)
    # timestamps: 包含三列：user id，item id，归一化时间戳
    # 可训练数据包括G_rating, G_timestamp, G_confidence, ratings，三个StellarGraph对象(不包含标签的图结构对象)和一个pandas data frame(包含标签的数据)
    '''

    @staticmethod
    def generator_TCAD_from_files(users, items, ratings, edges, timestamps, path="OUTPUT.TCAD"):
        # 创建StellarGraph对象
        G_rating = StellarGraph(nodes={'source': users, 'target': items},
                                edges={'weight': edges})
        G_timestamp = StellarGraph(nodes={'source': users, 'target': items},
                                   edges={'weight': timestamps})
        # 数据检查
        G_rating.check_graph_for_ml()
        G_timestamp.check_graph_for_ml()

        # 计算置信度的函数
        def confidence_cal(t, d, alpha=10):
            return math.exp(-alpha * t / d)

        # 生成置信度图对象
        user_nodes = G_timestamp.nodes(node_type='source')
        item_nodes = G_timestamp.nodes(node_type='target')
        degrees = G_timestamp.node_degrees()
        confidence = []
        source = []
        target = []
        for user in user_nodes:
            for item in G_timestamp.neighbors(user, include_edge_weight=True):
                degree_i = degrees[item.node]  # 邻居节点的度
                degree_u = degrees[user]  # 自己的度
                source.append(user)
                target.append(item.node)
                # confidence.append(degree / (MAX_TIMESTAMP - item.weight))
                confidence.append(confidence_cal(item.weight, degree_i) + confidence_cal(item.weight, degree_u) / 2)

        df_confidence_edges = pd.DataFrame({'source': source, 'target': target, 'weight': confidence})
        print(df_confidence_edges)

        G_confidence = StellarGraph(nodes={'source': users, 'target': items},
                                    edges={'weight': df_confidence_edges})
        # 数据检查
        G_confidence.check_graph_for_ml()
        # 数据对象序列化存档
        with open(file=path, mode='wb') as fw:
            pickle.dump(obj=(G_rating, G_timestamp, G_confidence, ratings), file=fw)
        print('Data has been generated successfully.')

        return G_rating, G_timestamp, G_confidence, ratings

    # ↓amazon-books数据集
    '''
    num_edges仅在mini=True时生效，表示选取数据集的边数
    '''

    @staticmethod
    @deprecated(version='1.0', reason="The Function will be deprecated")
    def books(mini=False, num_edges=SIZE_MINI_MIN):
        data_name = 'BOOKS'
        # users/items 相关常量
        max_len_user = 21
        max_len_book = 11
        # 数字字符和大写字母映射成数字(0-35)
        chars_dict = {chr(i): i - 55 for i in range(65, 91)}
        chars_dict.update({chr(i): i - 48 for i in range(48, 58)})
        temp_path = 'data/amazon-books/'
        if mini is True:
            chunk_size = 50
            source_url = temp_path + 'ratings.csv'
            mini_url = temp_path + 'ratings_mini.csv'
            url = temp_path + 'ratings_mini_clean.temp'
            output = temp_path + data_name + '_MINI.TAD'
            if os.access(mini_url, os.F_OK) is False:
                reader = pd.read_csv(source_url, header=None, chunksize=num_edges)
                reader.get_chunk().to_csv(mini_url, sep=',', encoding='utf8', index=False, header=False)
                source_url = mini_url
                reader.close()
        else:
            chunk_size = 200
            source_url = temp_path + 'ratings.csv'
            url = temp_path + 'ratings_clean.temp'
            output = temp_path + data_name + '.TAD'
        # 监测本地是否有相关数据序列化对象
        if os.access(output, os.F_OK):
            print('Serialized data object detected: \n"' + output + '"')
            with open(file=output, mode='rb') as fr:
                return pickle.load(file=fr)
        else:
            print('Serialized data object were not detected: \nData preprocessing...')

        # 归一化规则
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else 0 * x

        # 提交异步任务的工具方法
        collect = defaultdict(list)

        async def submitor(chunk, index, opera, name='res'):  # 只有一个返回值

            future = await opera(chunk, index)
            collect[name].append(future)

        async def submitor2(chunk, index, opera, name=None):  # 两个返回值
            if name is None:
                name = ['res1', 'res2']
            future1, future2 = await opera(chunk, index)
            collect[name[0]].append(future1)
            collect[name[1]].append(future2)

        # 方法：异步清理数据
        async def data_clean(chunk, index):
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            # print(chunk)
            return chunk

        # 步骤0：清理数据
        print("[Step 0] Data cleaning...")
        if os.access(url, os.F_OK):
            print('Temp detected: \n"' + url + '"')
        else:
            reader = pd.read_csv(source_url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, data_clean, 'data') for index, chunk in
                                  enumerate(reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 0] Data cleaning done.")
            print("[Step 0] Clean data writing back...")
            df = pd.concat(collect['data'], ignore_index=False)
            df.to_csv(url, sep=',', encoding='utf8', index=False, header=False)
            print("[Step 0] Clean data writing back done.")
            reader.close()
            # print(df_users)

        # 方法：异步生成chunk用户特征
        async def get_user_features(chunk, index):
            # print("[Step 1/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            for i in range(max_len_user):
                chunk[str(i)] = -1
            for i in range(max_len_user):
                chunk[str(i)] = chunk['user'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['user']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 方法：异步生成chunk项目特征
        async def get_item_features(chunk, index):
            # print("[Step 2/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            for i in range(max_len_book):
                chunk[str(i)] = -1
            for i in range(max_len_book):
                chunk[str(i)] = chunk['item'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['item']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 步骤1：分chunk提交异步任务生成users特征
        print("[Step 1] Users features generating...")
        if os.access(temp_path + 'users.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'users.temp"')
            df_users = pd.read_table(temp_path + 'users.temp', encoding='utf8', sep='\t', index_col='user')
            # print(df_users)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_user_features, 'user_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            # print(collect['user_features'])
            print("[Step 1] Users features generating done.")
            print("[Step 1] Users features writing back...")
            df_users = pd.concat(collect['user_features'], ignore_index=False)
            df_users = df_users.drop_duplicates()
            df_users = df_users.apply(max_min_scaler)
            df_users.to_csv(temp_path + 'users.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 1] Users features writing back done.")
            ratings_reader.close()
            # print(df_users)

        # 步骤2：分chunk提交异步任务生成books特征
        print("[Step 2] Items features generating...")
        if os.access(temp_path + 'items.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'items.temp"')
            df_items = pd.read_table(temp_path + 'items.temp', encoding='utf8', sep='\t', index_col='item')
            # print(df_books)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_item_features, 'item_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 2] Items features generating done.")
            print("[Step 2] Items features writing back...")
            df_items = pd.concat(collect['item_features'], ignore_index=False)
            df_items = df_items.drop_duplicates()
            df_items = df_items.apply(max_min_scaler)
            df_items.to_csv(temp_path + 'items.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 2] Items features writing back done.")
            ratings_reader.close()
            # print(df_books)

        # ratings/timestamps
        # 方法：异步生成ratings矩阵
        async def get_ratings_edges(chunk, index):
            # print("[Step 3/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['timestamp']
            chunk.columns = ['source', 'target', 'weight']
            edges = chunk.copy(deep=True)
            edges['weight'] = 1
            # print(edges)
            return chunk, edges

        # 方法：异步生成ratings矩阵
        async def get_timestamps(chunk, index):
            # print("[Step 4/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['rating']
            chunk.columns = ['source', 'target', 'weight']
            # print(chunk)
            return chunk

        # 步骤：分chunk异步生成ratings/edges
        print("[Step 3] Ratings and Edges generating...")
        if os.access(temp_path + 'ratings.temp', os.F_OK) and os.access(
                temp_path + 'edges.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'ratings.temp"\n"' + temp_path + 'ratings.temp"')
            df_ratings = pd.read_table(temp_path + 'ratings.temp', encoding='utf8', sep='\t', index_col=0)
            df_edges = pd.read_table(temp_path + 'edges.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor2(chunk, index, get_ratings_edges, ['ratings', 'edges']) for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 3] Ratings and Edges generating done.")
            print("[Step 3] Ratings and Edges writing back...")
            df_ratings = pd.concat(collect['ratings'], ignore_index=False)
            df_edges = pd.concat(collect['edges'], ignore_index=False)
            df_ratings.to_csv(temp_path + 'ratings.temp', sep='\t', encoding='utf8', index=True)
            df_edges.to_csv(temp_path + 'edges.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 3] Ratings and Edges writing back done;")
            ratings_reader.close()

        # 步骤：分chunk异步生成timestamps
        print("[Step 4] Timestamps generating...")
        if os.access(temp_path + 'timestamps.temp', os.F_OK):
            print('Temp detected: \n' + temp_path + '"timestamps.temp"')
            df_timestamps = pd.read_table(temp_path + 'timestamps.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait(
                [submitor(chunk, index, get_timestamps, 'timestamps') for index, chunk in
                 enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 4] Timestamps generating done.")
            print("[Step 4] Timestamps writing back...")
            df_timestamps = pd.concat(collect['timestamps'], ignore_index=False)
            df_timestamps['weight'] = df_timestamps[['weight']].apply(max_min_scaler)
            df_timestamps.to_csv(temp_path + 'timestamps.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 4] Timestamps writing back done.")
            ratings_reader.close()

        # 创建StellarGraph对象
        # print(df_users)
        print(df_edges)
        G_rating = StellarGraph(nodes={'source': df_users, 'target': df_items},
                                edges={'weight': df_edges})
        G_timestamp = StellarGraph(nodes={'source': df_users, 'target': df_items},
                                   edges={'weight': df_timestamps})

        # 数据检查
        G_rating.check_graph_for_ml()
        G_timestamp.check_graph_for_ml()

        # 数据对象序列化存档
        with open(file=output, mode='wb') as fw:
            pickle.dump(obj=(G_rating, G_timestamp, df_ratings), file=fw)
        print('Data has been preprocessed successfully.')
        return G_rating, G_timestamp, df_ratings

    # ↓amazon-cds数据集
    '''
    num_edges仅在mini=True时生效，表示选取数据集的边数
    '''

    @staticmethod
    @deprecated(version='1.0', reason="The Function will be deprecated")
    def cds(mini=False, num_edges=SIZE_MINI_MIN):
        data_name = 'CDS'
        # users/items 相关常量
        max_len_user = 21
        max_len_book = 11
        # 数字字符和大写字母映射成数字(0-35)
        chars_dict = {chr(i): i - 55 for i in range(65, 91)}
        chars_dict.update({chr(i): i - 48 for i in range(48, 58)})
        temp_path = 'data/amazon-cds/'
        if mini is True:
            chunk_size = 50
            source_url = temp_path + 'ratings.csv'
            mini_url = temp_path + 'ratings_mini.csv'
            url = temp_path + 'ratings_mini_clean.temp'
            output = temp_path + data_name + '_MINI.TAD'
            if os.access(mini_url, os.F_OK) is False:
                reader = pd.read_csv(source_url, header=None, chunksize=num_edges)
                reader.get_chunk().to_csv(mini_url, sep=',', encoding='utf8', index=False, header=False)
                source_url = mini_url
                reader.close()
        else:
            chunk_size = 200
            source_url = temp_path + 'ratings.csv'
            url = temp_path + 'ratings_clean.temp'
            output = temp_path + data_name + '.TAD'
        # 监测本地是否有相关数据序列化对象
        if os.access(output, os.F_OK):
            print('Serialized data object detected: \n"' + output + '"')
            with open(file=output, mode='rb') as fr:
                return pickle.load(file=fr)
        else:
            print('Serialized data object were not detected: \nData preprocessing...')

        # 归一化规则
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else 0 * x

        # 提交异步任务的工具方法
        collect = defaultdict(list)

        async def submitor(chunk, index, opera, name='res'):  # 只有一个返回值

            future = await opera(chunk, index)
            collect[name].append(future)

        async def submitor2(chunk, index, opera, name=None):  # 两个返回值
            if name is None:
                name = ['res1', 'res2']
            future1, future2 = await opera(chunk, index)
            collect[name[0]].append(future1)
            collect[name[1]].append(future2)

        # 方法：异步清理数据
        async def data_clean(chunk, index):
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            # print(chunk)
            return chunk

        # 步骤0：清理数据
        print("[Step 0] Data cleaning...")
        if os.access(url, os.F_OK):
            print('Temp detected: \n"' + url + '"')
        else:
            reader = pd.read_csv(source_url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, data_clean, 'data') for index, chunk in
                                  enumerate(reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 0] Data cleaning done.")
            print("[Step 0] Clean data writing back...")
            df = pd.concat(collect['data'], ignore_index=False)
            df.to_csv(url, sep=',', encoding='utf8', index=False, header=False)
            print("[Step 0] Clean data writing back done.")
            reader.close()
            # print(df_users)

        # 方法：异步生成chunk用户特征
        async def get_user_features(chunk, index):
            # print("[Step 1/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            for i in range(max_len_user):
                chunk[str(i)] = -1
            for i in range(max_len_user):
                chunk[str(i)] = chunk['user'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['user']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 方法：异步生成chunk项目特征
        async def get_item_features(chunk, index):
            # print("[Step 2/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            for i in range(max_len_book):
                chunk[str(i)] = -1
            for i in range(max_len_book):
                chunk[str(i)] = chunk['item'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['item']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 步骤1：分chunk提交异步任务生成users特征
        print("[Step 1] Users features generating...")
        if os.access(temp_path + 'users.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'users.temp"')
            df_users = pd.read_table(temp_path + 'users.temp', encoding='utf8', sep='\t', index_col='user')
            # print(df_users)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_user_features, 'user_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            # print(collect['user_features'])
            print("[Step 1] Users features generating done.")
            print("[Step 1] Users features writing back...")
            df_users = pd.concat(collect['user_features'], ignore_index=False)
            df_users = df_users.drop_duplicates()
            df_users = df_users.apply(max_min_scaler)
            df_users.to_csv(temp_path + 'users.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 1] Users features writing back done.")
            ratings_reader.close()
            # print(df_users)

        # 步骤2：分chunk提交异步任务生成items特征
        print("[Step 2] Items features generating...")
        if os.access(temp_path + 'items.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'items.temp"')
            df_items = pd.read_table(temp_path + 'items.temp', encoding='utf8', sep='\t', index_col='item')
            # print(df_items)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_item_features, 'item_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 2] Items features generating done.")
            print("[Step 2] Items features writing back...")
            df_items = pd.concat(collect['item_features'], ignore_index=False)
            df_items = df_items.drop_duplicates()
            df_items = df_items.apply(max_min_scaler)
            df_items.to_csv(temp_path + 'items.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 2] Items features writing back done.")
            # ratings_reader.close()
            # print(df_items)

        # ratings/timestamps
        # 方法：异步生成ratings矩阵
        async def get_ratings_edges(chunk, index):
            # print("[Step 3/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['timestamp']
            chunk.columns = ['source', 'target', 'weight']
            edges = chunk.copy(deep=True)
            edges['weight'] = 1
            # print(edges)
            return chunk, edges

        # 方法：异步生成ratings矩阵
        async def get_timestamps(chunk, index):
            # print("[Step 4/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['rating']
            chunk.columns = ['source', 'target', 'weight']
            # print(chunk)
            return chunk

        # 步骤：分chunk异步生成ratings/edges
        print("[Step 3] Ratings and Edges generating...")
        if os.access(temp_path + 'ratings.temp', os.F_OK) and os.access(
                temp_path + 'edges.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'ratings.temp"\n"' + temp_path + 'edges.temp"')
            df_ratings = pd.read_table(temp_path + 'ratings.temp', encoding='utf8', sep='\t', index_col=0)
            df_edges = pd.read_table(temp_path + 'edges.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor2(chunk, index, get_ratings_edges, ['ratings', 'edges']) for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 3] Ratings and Edges generating done.")
            print("[Step 3] Ratings and Edges writing back...")
            df_ratings = pd.concat(collect['ratings'], ignore_index=False)
            df_edges = pd.concat(collect['edges'], ignore_index=False)
            df_ratings.to_csv(temp_path + 'ratings.temp', sep='\t', encoding='utf8', index=True)
            df_edges.to_csv(temp_path + 'edges.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 3] Ratings and Edges writing back done;")
            ratings_reader.close()

        # 步骤：分chunk异步生成timestamps
        print("[Step 4] Timestamps generating...")
        if os.access(temp_path + 'timestamps.temp', os.F_OK):
            print('Temp detected: \n' + temp_path + '"timestamps.temp"')
            df_timestamps = pd.read_table(temp_path + 'timestamps.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait(
                [submitor(chunk, index, get_timestamps, 'timestamps') for index, chunk in
                 enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 4] Timestamps generating done.")
            print("[Step 4] Timestamps writing back...")
            df_timestamps = pd.concat(collect['timestamps'], ignore_index=False)
            df_timestamps['weight'] = df_timestamps[['weight']].apply(max_min_scaler)
            df_timestamps.to_csv(temp_path + 'timestamps.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 4] Timestamps writing back done.")
            ratings_reader.close()

        # 创建StellarGraph对象
        # print(df_users)
        print(df_edges)
        G_rating = StellarGraph(nodes={'source': df_users, 'target': df_items},
                                edges={'weight': df_edges})
        G_timestamp = StellarGraph(nodes={'source': df_users, 'target': df_items},
                                   edges={'weight': df_timestamps})

        # 数据检查
        G_rating.check_graph_for_ml()
        G_timestamp.check_graph_for_ml()

        # 数据对象序列化存档
        with open(file=output, mode='wb') as fw:
            pickle.dump(obj=(G_rating, G_timestamp, df_ratings), file=fw)
        print('Data has been preprocessed successfully.')
        return G_rating, G_timestamp, df_ratings

    # ↓获取Amazon数据集
    '''
    # 该方法是对books、cds两个方法进一步封装
    # name in {'amazon-books','amazon-cds','amazon-foods','amazon-toys','amazon-musics', 'amazon-movies'}
    # num_edges仅在mini=True时生效，表示选取数据集的边数
    '''

    def amazon(name, mini=False, num_edges=SIZE_MINI_MIN):
        print(
            "If an error occurs in the last step of data processing, "
            "it may be caused by a memory allocation failure when merging the results of batch tasks. "
            "Please do not delete the cache and try again. If you still can’t solve it, "
            "there may be a problem with the source data set itself, "
            "please confirm that your source data is accurate.")
        data_list = ['amazon-books', 'amazon-cds', 'amazon-foods', 'amazon-toys', 'amazon-musics', 'amazon-movies']
        if name not in data_list:
            print(
                "Please make sure that dataset name is correct.\nHere are the amazon dataset now we provided:\n" + str(
                    data_list))
            exit(0)
        data_name = str(name).split('-')[1].upper()
        # users/items 相关常量
        max_len_user = 21
        max_len_book = 11
        # 数字字符和大写字母映射成数字(0-35)
        chars_dict = {chr(i): i - 55 for i in range(65, 91)}
        chars_dict.update({chr(i): i - 48 for i in range(48, 58)})
        temp_path = 'data/' + str(name) + '/'
        if mini is True:
            chunk_size = 50
            source_url = temp_path + 'ratings.csv'
            mini_url = temp_path + 'ratings_mini.csv'
            url = temp_path + 'ratings_mini_clean.temp'
            output = temp_path + data_name + '_MINI.TAD'
            if os.access(mini_url, os.F_OK) is False:
                reader = pd.read_csv(source_url, header=None, chunksize=num_edges)
                reader.get_chunk().to_csv(mini_url, sep=',', encoding='utf8', index=False, header=False)
                source_url = mini_url
                reader.close()
        else:
            chunk_size = 200
            source_url = temp_path + 'ratings.csv'
            url = temp_path + 'ratings_clean.temp'
            output = temp_path + data_name + '.TAD'
        # 监测本地是否有相关数据序列化对象
        if os.access(output, os.F_OK):
            print('Serialized data object detected: \n"' + output + '"')
            with open(file=output, mode='rb') as fr:
                return pickle.load(file=fr)
        else:
            print('Serialized data object were not detected: \nData preprocessing...')

        # 归一化规则
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else 0 * x

        # 提交异步任务的工具方法
        collect = defaultdict(list)

        async def submitor(chunk, index, opera, name='res'):  # 只有一个返回值

            future = await opera(chunk, index)
            collect[name].append(future)

        async def submitor2(chunk, index, opera, name=None):  # 两个返回值
            if name is None:
                name = ['res1', 'res2']
            future1, future2 = await opera(chunk, index)
            collect[name[0]].append(future1)
            collect[name[1]].append(future2)

        # 方法：异步清理数据
        async def data_clean(chunk, index):
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            # print(chunk)
            return chunk

        # 步骤0：清理数据
        print("[Step 0] Data cleaning...")
        if os.access(url, os.F_OK):
            print('Temp detected: \n"' + url + '"')
        else:
            reader = pd.read_csv(source_url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, data_clean, 'data') for index, chunk in
                                  enumerate(reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 0] Data cleaning done.")
            print("[Step 0] Clean data writing back...")
            df = pd.concat(collect['data'], ignore_index=False)
            df.to_csv(url, sep=',', encoding='utf8', index=False, header=False)
            print("[Step 0] Clean data writing back done.")
            reader.close()
            # print(df_users)

        # 方法：异步生成chunk用户特征
        async def get_user_features(chunk, index):
            # print("[Step 1/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            for i in range(max_len_user):
                chunk[str(i)] = -1
            for i in range(max_len_user):
                chunk[str(i)] = chunk['user'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['user']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 方法：异步生成chunk项目特征
        async def get_item_features(chunk, index):
            # print("[Step 2/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            for i in range(max_len_book):
                chunk[str(i)] = -1
            for i in range(max_len_book):
                chunk[str(i)] = chunk['item'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['item']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 步骤1：分chunk提交异步任务生成users特征
        print("[Step 1] Users features generating...")
        if os.access(temp_path + 'users.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'users.temp"')
            df_users = pd.read_table(temp_path + 'users.temp', encoding='utf8', sep='\t', index_col='user')
            # print(df_users)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_user_features, 'user_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            # print(collect['user_features'])
            print("[Step 1] Users features generating done.")
            print("[Step 1] Users features writing back...")
            df_users = pd.concat(collect['user_features'], ignore_index=False)
            df_users = df_users.drop_duplicates()
            df_users = df_users.apply(max_min_scaler)
            df_users.to_csv(temp_path + 'users.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 1] Users features writing back done.")
            ratings_reader.close()
            # print(df_users)

        # 步骤2：分chunk提交异步任务生成books特征
        print("[Step 2] Items features generating...")
        if os.access(temp_path + 'items.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'items.temp"')
            df_items = pd.read_table(temp_path + 'items.temp', encoding='utf8', sep='\t', index_col='item')
            # print(df_books)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_item_features, 'item_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 2] Items features generating done.")
            print("[Step 2] Items features writing back...")
            df_items = pd.concat(collect['item_features'], ignore_index=False)
            df_items = df_items.drop_duplicates()
            df_items = df_items.apply(max_min_scaler)
            df_items.to_csv(temp_path + 'items.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 2] Items features writing back done.")
            ratings_reader.close()
            # print(df_books)

        # ratings/timestamps
        # 方法：异步生成ratings矩阵
        async def get_ratings_edges(chunk, index):
            # print("[Step 3/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['timestamp']
            chunk.columns = ['source', 'target', 'weight']
            edges = chunk.copy(deep=True)
            edges['weight'] = 1
            # print(edges)
            return chunk, edges

        # 方法：异步生成ratings矩阵
        async def get_timestamps(chunk, index):
            # print("[Step 4/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['rating']
            chunk.columns = ['source', 'target', 'weight']
            # print(chunk)
            return chunk

        # 步骤：分chunk异步生成ratings/edges
        print("[Step 3] Ratings and Edges generating...")
        if os.access(temp_path + 'ratings.temp', os.F_OK) and os.access(
                temp_path + 'edges.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'ratings.temp"\n"' + temp_path + 'ratings.temp"')
            df_ratings = pd.read_table(temp_path + 'ratings.temp', encoding='utf8', sep='\t', index_col=0)
            df_edges = pd.read_table(temp_path + 'edges.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor2(chunk, index, get_ratings_edges, ['ratings', 'edges']) for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 3] Ratings and Edges generating done.")
            print("[Step 3] Ratings and Edges writing back...")
            df_ratings = pd.concat(collect['ratings'], ignore_index=False)
            df_edges = pd.concat(collect['edges'], ignore_index=False)
            df_ratings.to_csv(temp_path + 'ratings.temp', sep='\t', encoding='utf8', index=True)
            df_edges.to_csv(temp_path + 'edges.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 3] Ratings and Edges writing back done;")
            ratings_reader.close()

        # 步骤：分chunk异步生成timestamps
        print("[Step 4] Timestamps generating...")
        if os.access(temp_path + 'timestamps.temp', os.F_OK):
            print('Temp detected: \n' + temp_path + '"timestamps.temp"')
            df_timestamps = pd.read_table(temp_path + 'timestamps.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait(
                [submitor(chunk, index, get_timestamps, 'timestamps') for index, chunk in
                 enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 4] Timestamps generating done.")
            print("[Step 4] Timestamps writing back...")
            df_timestamps = pd.concat(collect['timestamps'], ignore_index=False)
            df_timestamps['weight'] = df_timestamps[['weight']].apply(max_min_scaler)
            df_timestamps.to_csv(temp_path + 'timestamps.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 4] Timestamps writing back done.")
            ratings_reader.close()

        # 创建StellarGraph对象
        # print(df_users)
        print(df_edges)
        G_rating = StellarGraph(nodes={'source': df_users, 'target': df_items},
                                edges={'weight': df_edges})
        G_timestamp = StellarGraph(nodes={'source': df_users, 'target': df_items},
                                   edges={'weight': df_timestamps})

        # 数据检查
        G_rating.check_graph_for_ml()
        G_timestamp.check_graph_for_ml()

        # 数据对象序列化存档
        with open(file=output, mode='wb') as fw:
            pickle.dump(obj=(G_rating, G_timestamp, df_ratings), file=fw)
        print('Data has been preprocessed successfully.')
        return G_rating, G_timestamp, df_ratings

    # ↓获取Amazon数据集(包含置信度图结构)
    '''
    # 该方法是对books、cds两个方法进一步封装
    # name in {'amazon-books','amazon-cds','amazon-foods','amazon-toys','amazon-musics', 'amazon-movies'}
    # num_edges仅在mini=True时生效，表示选取数据集的边数
    '''

    def amazon_c(name, mini=False, num_edges=10000):
        print(
            "If an error occurs in the last step of data processing, it may be caused by a memory allocation failure when merging the results of batch tasks. Please do not delete the cache and try again. If you still can’t solve it, there may be a problem with the source data set itself, please confirm that your source data is accurate.")
        data_list = ['amazon-books', 'amazon-cds', 'amazon-foods', 'amazon-toys', 'amazon-musics', 'amazon-movies']
        if name not in data_list:
            print(
                "Please make sure that dataset name is correct.\nHere are the amazon dataset now we provided:\n" + str(
                    data_list))
            exit(0)
        data_name = str(name).split('-')[1].upper()
        # users/items 相关常量
        max_len_user = 21
        max_len_book = 11
        # 数字字符和大写字母映射成数字(0-35)
        chars_dict = {chr(i): i - 55 for i in range(65, 91)}
        chars_dict.update({chr(i): i - 48 for i in range(48, 58)})
        temp_path = 'data/' + str(name) + '/'
        if mini is True:
            chunk_size = 50
            source_url = temp_path + 'ratings.csv'
            mini_url = temp_path + 'ratings_mini.csv'
            url = temp_path + 'ratings_mini_clean.temp'
            output = temp_path + data_name + '_MINI.TCAD'
            if os.access(mini_url, os.F_OK) is False:
                reader = pd.read_csv(source_url, header=None, chunksize=num_edges)
                reader.get_chunk().to_csv(mini_url, sep=',', encoding='utf8', index=False, header=False)
                source_url = mini_url
                reader.close()
        else:
            chunk_size = 200
            source_url = temp_path + 'ratings.csv'
            url = temp_path + 'ratings_clean.temp'
            output = temp_path + data_name + '.TCAD'
        # 监测本地是否有相关数据序列化对象
        if os.access(output, os.F_OK):
            print('Serialized data object detected: \n"' + output + '"')
            with open(file=output, mode='rb') as fr:
                return pickle.load(file=fr)
        else:
            print('Serialized data object were not detected: \nData preprocessing...')

        # 归一化规则
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if (np.max(x) - np.min(x)) != 0 else 0 * x

        # 提交异步任务的工具方法
        collect = defaultdict(list)

        async def submitor(chunk, index, opera, name='res'):  # 只有一个返回值

            future = await opera(chunk, index)
            collect[name].append(future)

        async def submitor2(chunk, index, opera, name=None):  # 两个返回值
            if name is None:
                name = ['res1', 'res2']
            future1, future2 = await opera(chunk, index)
            collect[name[0]].append(future1)
            collect[name[1]].append(future2)

        # 方法：异步清理数据
        async def data_clean(chunk, index):
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            # print(chunk)
            return chunk

        # 步骤0：清理数据
        print("[Step 0] Data cleaning...")
        if os.access(url, os.F_OK):
            print('Temp detected: \n"' + url + '"')
        else:
            reader = pd.read_csv(source_url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, data_clean, 'data') for index, chunk in
                                  enumerate(reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 0] Data cleaning done.")
            print("[Step 0] Clean data writing back...")
            df = pd.concat(collect['data'], ignore_index=False)
            df.to_csv(url, sep=',', encoding='utf8', index=False, header=False)
            print("[Step 0] Clean data writing back done.")
            reader.close()
            # print(df_users)

        # 方法：异步生成chunk用户特征
        async def get_user_features(chunk, index):
            # print("[Step 1/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            chunk['user'] = chunk['user'].apply(str).apply(lambda x: x.replace("?", ""))
            for i in range(max_len_user):
                chunk[str(i)] = -1
            for i in range(max_len_user):
                chunk[str(i)] = chunk['user'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['user']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 方法：异步生成chunk项目特征
        async def get_item_features(chunk, index):
            # print("[Step 2/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            for i in range(max_len_book):
                chunk[str(i)] = -1
            for i in range(max_len_book):
                chunk[str(i)] = chunk['item'].apply(str).apply(
                    lambda x: chars_dict[x[i]] if i < len(x) and x[i] in chars_dict.keys() else -1)
            chunk.index = chunk['item']
            del chunk['user']
            del chunk['item']
            del chunk['rating']
            del chunk['timestamp']
            # print(chunk)
            return chunk

        # 步骤1：分chunk提交异步任务生成users特征
        print("[Step 1] Users features generating...")
        if os.access(temp_path + 'users.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'users.temp"')
            df_users = pd.read_table(temp_path + 'users.temp', encoding='utf8', sep='\t', index_col='user')
            # print(df_users)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_user_features, 'user_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            # print(collect['user_features'])
            print("[Step 1] Users features generating done.")
            print("[Step 1] Users features writing back...")
            df_users = pd.concat(collect['user_features'], ignore_index=False)
            df_users = df_users.drop_duplicates()
            df_users = df_users.apply(max_min_scaler)
            df_users.to_csv(temp_path + 'users.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 1] Users features writing back done.")
            ratings_reader.close()
            # print(df_users)

        # 步骤2：分chunk提交异步任务生成books特征
        print("[Step 2] Items features generating...")
        if os.access(temp_path + 'items.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'items.temp"')
            df_items = pd.read_table(temp_path + 'items.temp', encoding='utf8', sep='\t', index_col='item')
            # print(df_books)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor(chunk, index, get_item_features, 'item_features') for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 2] Items features generating done.")
            print("[Step 2] Items features writing back...")
            df_items = pd.concat(collect['item_features'], ignore_index=False)
            df_items = df_items.drop_duplicates()
            df_items = df_items.apply(max_min_scaler)
            df_items.to_csv(temp_path + 'items.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 2] Items features writing back done.")
            ratings_reader.close()
            # print(df_books)

        # ratings/timestamps
        # 方法：异步生成ratings矩阵
        async def get_ratings_edges(chunk, index):
            # print("[Step 3/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['timestamp']
            chunk.columns = ['source', 'target', 'weight']
            edges = chunk.copy(deep=True)
            edges['weight'] = 1
            # print(edges)
            return chunk, edges

        # 方法：异步生成ratings矩阵
        async def get_timestamps(chunk, index):
            # print("[Step 4/4]chunk " + str(index) + " is processing...")
            chunk.columns = ['user', 'item', 'rating', 'timestamp']
            del chunk['rating']
            chunk.columns = ['source', 'target', 'weight']
            # print(chunk)
            return chunk

        # 步骤：分chunk异步生成ratings/edges
        print("[Step 3] Ratings and Edges generating...")
        if os.access(temp_path + 'ratings.temp', os.F_OK) and os.access(
                temp_path + 'edges.temp', os.F_OK):
            print('Temp detected: \n"' + temp_path + 'ratings.temp"\n"' + temp_path + 'ratings.temp"')
            df_ratings = pd.read_table(temp_path + 'ratings.temp', encoding='utf8', sep='\t', index_col=0)
            df_edges = pd.read_table(temp_path + 'edges.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait([submitor2(chunk, index, get_ratings_edges, ['ratings', 'edges']) for index, chunk in
                                  enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 3] Ratings and Edges generating done.")
            print("[Step 3] Ratings and Edges writing back...")
            df_ratings = pd.concat(collect['ratings'], ignore_index=False)
            df_edges = pd.concat(collect['edges'], ignore_index=False)
            df_ratings.to_csv(temp_path + 'ratings.temp', sep='\t', encoding='utf8', index=True)
            df_edges.to_csv(temp_path + 'edges.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 3] Ratings and Edges writing back done;")
            ratings_reader.close()

        # 步骤：分chunk异步生成timestamps
        print("[Step 4] Timestamps generating...")
        if os.access(temp_path + 'timestamps.temp', os.F_OK):
            print('Temp detected: \n' + temp_path + '"timestamps.temp"')
            df_timestamps = pd.read_table(temp_path + 'timestamps.temp', encoding='utf8', sep='\t', index_col=0)
        else:
            ratings_reader = pd.read_csv(url, header=None, chunksize=chunk_size)
            loop = asyncio.get_event_loop()
            tasks = asyncio.wait(
                [submitor(chunk, index, get_timestamps, 'timestamps') for index, chunk in
                 enumerate(ratings_reader, 1)])  # 划分chunk
            loop.run_until_complete(tasks)
            print("[Step 4] Timestamps generating done.")
            print("[Step 4] Timestamps writing back...")
            df_timestamps = pd.concat(collect['timestamps'], ignore_index=False)
            df_timestamps['weight'] = df_timestamps[['weight']].apply(max_min_scaler)
            df_timestamps.to_csv(temp_path + 'timestamps.temp', sep='\t', encoding='utf8', index=True)
            print("[Step 4] Timestamps writing back done.")
            ratings_reader.close()

        return TimeAwaredDataset.generator_TCAD_from_files(df_users, df_items, df_ratings, df_edges, df_timestamps,
                                                           path=output)

    # ↓gowalla数据集(该数据集有点问题)
    @staticmethod
    @deprecated(version='1.0', reason="There are some problems in the dataset.")
    def gowalla():
        # 监测本地是否有相关数据序列化对象
        if os.access('data/gowalla/GOWALLA.TAD', os.F_OK):
            print('Serialized data object detected: "data/gowalla/GOWALLA.TAD"')
            with open(file='data/gowalla/GOWALLA.TAD', mode='rb') as fr:
                return pickle.load(file=fr)
        else:
            print('Serialized data object were not detected: Data preprocessing...')

        # 归一化规则
        max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

        # 时间戳转换规则
        timestamp_transfer = lambda x: int(time.mktime(time.strptime(str(x), "%Y-%m-%dT%H:%M:%SZ")))

        # 数据预处理
        # 1.时间戳转换
        if os.access('data/gowalla/checkins_timestamps.cache', os.F_OK):
            print('Loading: "data/gowalla/checkins_timestamps.cache"')
            df_checkins_timestamps = pd.read_table('data/gowalla/checkins_timestamps.cache', encoding='utf8',
                                                   sep='\t')
        else:
            df_checkins = pd.read_table('data/gowalla/checkins.txt', header=None, encoding='utf8', sep='\t')

            df_checkins.columns = ['user', 'timestamp', 'latitude', 'longitude', 'location']

            df_checkins['timestamp'] = df_checkins['timestamp'].apply(timestamp_transfer)
            df_checkins_timestamps = df_checkins
            df_checkins_timestamps['user'] = 'u_' + df_checkins_timestamps['user'].apply(str)
            df_checkins_timestamps['location'] = 'l_' + df_checkins_timestamps['location'].apply(str)
            df_checkins_timestamps.to_csv('data/gowalla/checkins_timestamps.cache', sep='\t', encoding='utf8',
                                          index=False)
        # 2.checkins序列数据特征化
        # 2.1 users
        if os.access('data/gowalla/users.cache', os.F_OK):
            print('Loading: "data/gowalla/users.cache"')
            df_users = pd.read_table('data/gowalla/users.cache', encoding='utf8', sep='\t', index_col='user')
        else:
            users_groups = df_checkins_timestamps.groupby(['user'])
            df_users_timestamps_mean = users_groups['timestamp'].mean()
            df_users_timestamps_std = users_groups['timestamp'].std()
            df_users_latitudes_mean = users_groups['latitude'].mean()
            df_users_latitudes_std = users_groups['latitude'].std()
            df_users_longitudes_mean = users_groups['longitude'].mean()
            df_users_longitudes_std = users_groups['longitude'].std()
            df_users = pd.concat(
                [df_users_timestamps_mean, df_users_timestamps_std, df_users_latitudes_mean,
                 df_users_latitudes_std,
                 df_users_longitudes_mean, df_users_longitudes_std], axis=1)
            # df_users.index = df_users['user']
            df_users = df_users.fillna(0)
            df_users.columns = ['timestamp_mean', 'timestamp_std', 'latitude_mean', 'latitude_std',
                                'longitude_mean', 'longitude_std']
            df_users = df_users.apply(max_min_scaler)
            df_users.to_csv('data/gowalla/users.cache', sep='\t', encoding='utf8', index=True)
        print(df_users)
        # 2.2 locations
        if os.access('data/gowalla/locations.cache', os.F_OK):
            print('Loading: "data/gowalla/locations.cache"')
            df_locations = pd.read_table('data/gowalla/locations.cache', encoding='utf8', sep='\t',
                                         index_col='location')
        else:
            locations_groups = df_checkins_timestamps.groupby(['location'])
            df_locations_timestamps_mean = locations_groups['timestamp'].mean()
            df_locations_timestamps_std = locations_groups['timestamp'].std()
            df_locations_latitudes_mean = locations_groups['latitude'].mean()
            df_locations_longitudes_mean = locations_groups['longitude'].mean()
            df_locations = pd.concat(
                [df_locations_timestamps_mean, df_locations_timestamps_std,
                 df_locations_latitudes_mean, df_locations_longitudes_mean], axis=1)
            # df_users.index = df_users['user']
            df_locations = df_locations.fillna(0)
            df_locations.columns = ['timestamp_mean', 'timestamp_std', 'latitude_mean', 'longitude_mean']
            df_locations = df_locations.apply(max_min_scaler)
            df_locations.to_csv('data/gowalla/locations.cache', sep='\t', encoding='utf8', index=True)

        print(df_locations)
        # 3.边数据读取
        edges_groups = df_checkins_timestamps.groupby(['user', 'location'])
        df_timestamps_edges = edges_groups['timestamp'].max()
        df_checkins_edges = edges_groups['timestamp'].count()
        df_timestamps_edges = pd.concat([df_timestamps_edges], axis=1).reset_index()
        df_checkins_edges = pd.concat([df_checkins_edges], axis=1).reset_index()
        df_timestamps_edges.columns = ['source', 'target', 'weight']
        df_checkins_edges.columns = ['source', 'target', 'weight']
        df_timestamps_edges['weight'] = df_timestamps_edges[['weight']].apply(max_min_scaler)
        edges_with_checkins = df_checkins_edges.copy(deep=True)  # 保留一份包含评分(签到次数)的edges数据帧用作返回值（验证标签）
        df_checkins_edges['weight'] = 1  # 构建图对象就把权重都置为１（输入数据，避免标签泄露）
        print(df_timestamps_edges)
        print(df_checkins_edges)

        # 创建StellarGraph对象
        G_rating = StellarGraph(nodes={'source': df_users, 'target': df_locations},
                                edges={'weight': df_checkins_edges})
        G_timestamp = StellarGraph(nodes={'source': df_users, 'target': df_locations},
                                   edges={'weight': df_timestamps_edges})

        # 数据检查
        G_rating.check_graph_for_ml()
        G_timestamp.check_graph_for_ml()

        # 数据对象序列化存档
        with open(file='data/gowalla/GOWALLA.TAD', mode='wb') as fw:
            pickle.dump(obj=(G_rating, G_timestamp, df_checkins_edges), file=fw)

        return G_rating, G_timestamp, edges_with_checkins

    # datasource: baseline NGCF used TAD->txt
    @staticmethod
    def get_users_items_ratings(read_path, write_path):
        G1, G2, edges = TimeAwaredDataset.load(read_path)
        sources_num = min(G1.nodes("target", use_ilocs=True))
        source_features = G1.node_features(node_type="source")
        target_features = G1.node_features(node_type="target")
        ids_to_ilocs = G1.node_ids_to_ilocs(G1.nodes())
        ilocs_to_ids = G1.node_ilocs_to_ids(ids_to_ilocs)
        map_ilocs = {ilocs_to_ids[i]: ids_to_ilocs[i] for i in range(len(ilocs_to_ids))}
        edges['source'] = edges['source'].apply(lambda x: map_ilocs[x])
        edges['target'] = edges['target'].apply(lambda x: map_ilocs[x] - sources_num)
        ratings = np.array(edges, dtype=np.int64)
        np.savetxt(fname=write_path + '/sources.txt', X=source_features)
        np.savetxt(fname=write_path + '/targets.txt', X=target_features)
        np.savetxt(fname=write_path + '/ratings.txt', X=ratings, fmt='%i %i %i')
        return source_features, target_features, ratings

    @staticmethod
    def get_sparseness_of_ratings(X):
        user_num = max(X[:, 0]) + 1
        item_num = max(X[:, 1]) + 1
        num = user_num * item_num
        sparseness = 1 - X.shape[0] / num
        return sparseness

    @staticmethod
    def get_num_of_ratings(X):
        return X.shape[0]

    @staticmethod
    def get_usernum_from_ratings(X):
        return max(X[:, 0]) + 1

    @staticmethod
    def get_itemnum_from_ratings(X):
        return max(X[:, 1]) + 1

    # TAD->jpg
    @staticmethod
    def get_time_rating_distribute(read_path, write_path, groups=50):
        G1, G2, edges = TimeAwaredDataset.load(read_path)
        times = G2.edges(include_edge_weight=True)[1]
        plt.hist(times, bins=groups)
        plt.xlabel('Timestamps(normalized)')
        plt.ylabel('Rating Counts')
        plt.savefig(write_path + '/time_rating_distribute.jpg', bbox_inches='tight')
        plt.close()

    # TCAD->jpg
    @staticmethod
    def get_cofidence_rating_distribute(read_path, write_path, groups=50):
        G1, G2, G3, edges = TimeAwaredDataset.load(read_path)
        confidences = G3.edges(include_edge_weight=True)[1]
        MAX= max(confidences)
        MIN = min(confidences)
        confidences = [(confidence-MIN)/(MAX-MIN) for confidence in confidences]
        plt.hist(confidences, bins=groups)
        plt.xlabel('Confidences(normalized)')
        plt.ylabel('Rating Counts')
        plt.savefig(write_path + '/cofidence_rating_distribute.jpg', bbox_inches='tight')
        plt.close()

# TimeAwaredDataset.cache_clear('amazon-cds', del_result=False)
# TimeAwaredDataset.amazon('amazon-cds',mini=True,num_edges=500000)

# G, G2, edges_with_ratings = TimeAwaredDataset.amazon("amazon-cds")
# print(G.edges(include_edge_weight=True))
# G, G_t, G_c, edges_with_ratings = TimeAwaredDataset.amazon_c("amazon-musics")
# G, G_t, G_c, edges_with_ratings = TimeAwaredDataset.amazon_c("amazon-movies")
# G, G_t, G_c, edges_with_ratings = TimeAwaredDataset.amazon_c("amazon-foods")
# G, G_t, G_c, edges_with_ratings = TimeAwaredDataset.amazon_c("amazon-cds")
# G, G_t, edges_with_ratings = TimeAwaredDataset.amazon("amazon-books", mini=True, num_edges=1000000)
# G, G_t, G_c, edges_with_ratings = TimeAwaredDataset.amazon_c("amazon-cds", mini=True, num_edges=500000)

# PREPARE DATA FOR NGCF
# TimeAwaredDataset.get_users_items_ratings('data/amazon-cds/CDS_MINI.TAD', 'data/amazon-cds')
# TimeAwaredDataset.get_users_items_ratings('data/amazon-books/BOOKS_MINI.TAD', 'data/amazon-books')
# TimeAwaredDataset.get_users_items_ratings('data/amazon-cds/CDS_MINI.TAD', 'data/amazon-cds')


# matrix = np.loadtxt('data/amazon-books/ratings.txt', dtype=np.int64)
# print(TimeAwaredDataset.get_itemnum_from_ratings(matrix))
# matrix = np.loadtxt('data/amazon-cds/ratings.txt', dtype=np.int64)
# print(TimeAwaredDataset.get_itemnum_from_ratings(matrix))
# matrix = np.loadtxt('data/amazon-foods/ratings.txt', dtype=np.int64)
# print(TimeAwaredDataset.get_itemnum_from_ratings(matrix))
# matrix = np.loadtxt('data/amazon-movies/ratings.txt', dtype=np.int64)
# print(TimeAwaredDataset.get_itemnum_from_ratings(matrix))
# matrix = np.loadtxt('data/amazon-musics/ratings.txt', dtype=np.int64)
# print(TimeAwaredDataset.get_itemnum_from_ratings(matrix))
# matrix = np.loadtxt('data/amazon-toys/ratings.txt', dtype=np.int64)
# print(TimeAwaredDataset.get_itemnum_from_ratings(matrix))

# TimeAwaredDataset.get_cofidence_rating_distribute('data/amazon-cds/CDS_MINI.TCAD', 'data/amazon-cds')
# TimeAwaredDataset.get_time_rating_distribute('data/amazon-cds/CDS_MINI.TAD', 'data/amazon-cds')
# TimeAwaredDataset.get_cofidence_rating_distribute('data/amazon-foods/FOODS.TCAD', 'data/amazon-foods')
# TimeAwaredDataset.get_time_rating_distribute('data/amazon-foods/FOODS.TAD', 'data/amazon-foods')
# TimeAwaredDataset.get_cofidence_rating_distribute('data/amazon-movies/MOVIES.TCAD', 'data/amazon-movies')
# TimeAwaredDataset.get_time_rating_distribute('data/amazon-movies/MOVIES.TAD', 'data/amazon-movies')
# TimeAwaredDataset.get_cofidence_rating_distribute('data/amazon-musics/MUSICS.TCAD', 'data/amazon-musics')
# TimeAwaredDataset.get_time_rating_distribute('data/amazon-musics/MUSICS.TAD', 'data/amazon-musics')
# TimeAwaredDataset.get_cofidence_rating_distribute('data/amazon-toys/TOYS.TCAD', 'data/amazon-toys')
# TimeAwaredDataset.get_time_rating_distribute('data/amazon-toys/TOYS.TAD', 'data/amazon-toys')