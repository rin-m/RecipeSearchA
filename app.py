import sqlite3
import csv
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn import preprocessing
from datetime import datetime


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/resultA", methods = ["POST"])
def resultA():
    # テンプレートに結果を渡してレンダリング
    recommended_recipes = recommended_recipe_list()
    return render_template('results.html', recommended_recipes=recommended_recipes)

def mood_sql():
    # データベースに接続
    conn = sqlite3.connect("recipe.db")
    cursor = conn.cursor()

    # データベースからデータを取得
    cursor.execute('SELECT * FROM recipe')
    result = cursor.fetchall()

    # 取得したデータをリストに格納
    recipe_list = []
    for row in result:
        recipe_list.append(list(row))

    # データベース接続を閉じる
    cursor.close()
    conn.close()
    return recipe_list


def recommended_recipe_list():
    # データベースからデータを取得
    recipe_list = mood_sql()

    # レシピデータの気分と分類を取得し標準化を行う
    recipe_list_std, mood_values = preprocess_recipe_list(recipe_list)

    # POSTメソッドから取得した気分の値とレシピデータの気分の値のユークリッド距離を算出
    euclidean_distance_list = calculate_euclidean_distance(recipe_list_std, mood_values)

    # POSTメソッドから取得した気分の値とレシピデータの気分の値のユークリッド距離を算出
    # euclidean_distance_list = euclidean_distance(recipe_list)

    # ユークリッド距離の値のリストとレシピデータのリストを結合
    sorted_list_reciprocal, sorted_list = sort_by_distance(recipe_list, euclidean_distance_list)

    # ランキングスコアを正規化
    C = normalized_score_list = normalize_score(sorted_list_reciprocal)

    # 多様性を考慮したリランキングを行う    
    # 結果をcsvファイルに書き込む
    # リストをresultに格納
    result_list = []
    alpha = 0.5
    R = 20
    C = normalized_score_list.copy()
    recommended_recipe_list = greedy_reranking(C, R, alpha)
    result_list.append(recommended_recipe_list)
    
    write_csv(result_list)

    # α=0.5のときの結果を返す
    return result_list[0]
    

def preprocess_recipe_list(recipe_list):
    # レシピデータの気分のラベルを取得['mood', 'body', 'money', 'time']
    moods = recipe_list[0][4:8]

    # レシピデータの気分と分類を取得
    recipe_list_syusyoku = []
    recipe_list_syusai = []
    recipe_list_hukusai = []
    recipe_list_mood = []
    recipe_list_body = []
    recipe_list_money = []
    recipe_list_time = []

    # レシピデータの気分と分類をリストに格納(先頭行のラベルを除く)
    for row in recipe_list[1:]:
        recipe_list_syusyoku.append(float(list(row)[1]))  # 「主食」の行のみを抽出
        recipe_list_syusai.append(float(list(row)[2]))    # 「主菜」の行のみを抽出
        recipe_list_hukusai.append(float(list(row)[3]))   # 「副菜」の行のみを抽出
        recipe_list_mood.append(float(list(row)[4]))      # 「精神」の行のみを抽出
        recipe_list_body.append(float(list(row)[5]))      # 「身体」の行のみを抽出
        recipe_list_money.append(float(list(row)[6]))     # 「経済」の行のみを抽出
        recipe_list_time.append(float(list(row)[7]))      # 「時間」の行のみを抽出

    # レシピデータの時間の値を標準化
    recipe_list_time_std = preprocessing.scale(recipe_list_time)

    # レシピデータの気分の値を標準化した値をリストに格納
    recipe_moods_std = [recipe_list_mood, recipe_list_body, recipe_list_money, recipe_list_time_std]

    global hover_times
    hover_times = {}
    global file_counter
    file_counter += 1
    
    # POSTメソッドから取得した気分の値を取得
    mood_values = [float(request.form[mood]) for mood in moods]
    # お疲れの項目を反転
    mood_values[0] = mood_values[0] * -1

    # 節約の項目を反転
    mood_values[2] = mood_values[2] * -1
    # 時間の項目を反転
    mood_values[3] = mood_values[3] * -1

    return recipe_moods_std, mood_values


def calculate_euclidean_distance(recipe_moods_std, mood_values):
    squared_diff = []

    for mood_index, mood_value in enumerate(mood_values):
        if mood_value != 0:

            diff_mood = np.array(recipe_moods_std[mood_index]) - mood_value # レシピデータの気分とPOSTメソッドから取得したmoodの値の差を計算
            squared_diff_mood = np.square(diff_mood)                 # 差の2乗を計算
        else:
            # スライダーが0を選択しているときはユークリッド距離を計算しない
            squared_diff_mood = np.zeros(len(recipe_moods_std[0]), dtype=float)
        
        squared_diff.append(squared_diff_mood)
    
    # 各列の合計を求める
    squared_diff_columns_sum = np.sum(squared_diff, axis=0)

    # 各要素の平方根を求める
    euclidean_distance_list = np.sqrt(squared_diff_columns_sum)

    # スコアが0となるとき逆数変換したスコアがinfとなる問題を回避するため、0の値を最小値の半分に置換する
    # euclidean_distance_listの0より大きい最小値を取得
    min_euclidean_distance = np.min(euclidean_distance_list[np.nonzero(euclidean_distance_list)])

    # min_euclidean_distanceの値を半分にする
    min_euclidean_distance_half = min_euclidean_distance / 2

    # euclidean_distance_listの0をmin_euclidean_distance_halfに置換
    euclidean_distance_list = np.where(euclidean_distance_list==0, min_euclidean_distance_half, euclidean_distance_list)

    return euclidean_distance_list


def sort_by_distance(recipe_list, euclidean_distance_list):
    # 距離の値のリストとデータベースから取得したデータのリストを結合
    combined_list = []
    recipe_list_deleted_label = recipe_list[1:] # 先頭行のラベルを除く
    for row_num in range(len(recipe_list_deleted_label)):
        recipe_list_deleted_label[row_num].append(euclidean_distance_list[row_num])
        combined_list.append(recipe_list_deleted_label[row_num])

    # 距離に基づいてリストをソート(昇順)
    sorted_list = sorted(combined_list, key=lambda x: x[8])

    # ランキングスコアを逆数に変換
    sorted_list_reciprocal = sort_list_reciprocal(sorted_list)

    # 多様性リランキング前のスコアを保存
    sorted_list_no_reranking = sorted_list_reciprocal.copy()

    return sorted_list_reciprocal, sorted_list_no_reranking


# リストの要素の値を逆数にしてリストを返す
# 検索結果を降順にソートするために使用
def sort_list_reciprocal(sorted_list):
    sorted_list_reciprocal = []

    for i in range(len(sorted_list)):
        sorted_item_reciprocal = sorted_list[i][0:8]
        sorted_list_reciprocal.append(sorted_item_reciprocal)
        sorted_list_reciprocal[i].append(1/sorted_list[i][8])

    return sorted_list_reciprocal


# 多様性を考慮したリランキングを行う
# ユークリッド距離を計算したリストを返す
def distance(i, R):

    v1 = np.array(i[1:4])
    dis_list = []
    dis_average = 0

    for j in R:
        v2 = np.array(j[1:4])
        dis = np.sqrt(np.sum(np.square(v1 - v2), axis=0))
        dis_list.append(dis)

    dis_average = np.average(dis_list)
    return dis_average


# 検索結果のリストを多様性を考慮してリランキング
def select_item_id(sorted_list, R, alpha):
    C = sorted_list
    max_score = -10
    max_score_item_id = -10
    for i in C:
        if len(R) == 0:
            return i

        score = alpha * i[8] + (1 - alpha) * distance(i, R)

        if score > max_score:
            max_score = score
            max_score_item_id = i
    return max_score_item_id


# 多様性を考慮したリランキングを行う
def greedy_reranking(C, N, alpha):
    results = []
    while len(results) < N:
        doc_id = select_item_id(C, results, alpha)
        results.append(doc_id)
        C.remove(doc_id)
        
    return results


# リストをcsvファイルに書き込む
def write_csv(list):
    DATE = datetime.now().strftime("%Y%m%d_%H%M%S")
    FILE_NAME = "./result/A_data_"+DATE+".csv"
    with open(FILE_NAME, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in list:
            writer.writerows(row)
            writer.writerow([])
        f.close()


# リストのスコアを正規化する
def normalize_score(search_result_candidate_list):
    # スコアだけを抽出して別のリストに格納
    list_score = []
    for row in search_result_candidate_list:
        list_score.append(row[8])

    # スコアを正規化
    list_score = preprocessing.minmax_scale(list_score, feature_range=(0, 1))

    # 正規化したスコアをリストに戻す
    normalized_score_list = []
    for i in range(len(search_result_candidate_list)):
        normalized_score_list.append(search_result_candidate_list[i][0:8])
        normalized_score_list[i].append(list_score[i])

    return normalized_score_list

# ファイルのカウンターを初期化
file_counter = 0

# /update_hover_time ルート
@app.route('/update_hover_time', methods=['POST'])
def update_hover_time():

    # ルート関数内でリクエストやセッションにアクセスする
    # (これはリクエストのコンテキスト内です)
    data = request.get_json()
    recipe_name = data.get('recipe')
    hover_time = data.get('time')

    hover_times[recipe_name] = hover_time

    # CSV ファイルにデータを保存
    save_hover_time_to_csv(hover_times)

    return jsonify(success=True)

def save_hover_time_to_csv(hover_times):
    global file_counter
    csv_filename = get_current_filename('./result/A_hover_times_', 'csv')

    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['recipe_name', 'hover_time'])  # ヘッダーを書き込む
        for recipe_name, hover_time in hover_times.items():
            writer.writerow([recipe_name, hover_time])

    print(f'Data saved to {csv_filename}')

def get_current_filename(base_name, extension):
    global file_counter
    filename = f"{base_name}_{file_counter}.{extension}"
    return filename

if __name__ == "__main__":
    app.run(debug=True, port=5000)