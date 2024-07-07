import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# 統計データとパーソナライズドデータの統合
statistical_data = {
    'age': [25, 35, 45, 55, 65],
    'gender': [0, 1, 0, 1, 0],  # 0: Male, 1: Female
    'region': [1, 2, 1, 2, 1],  # 1: Urban, 2: Rural
    'economic_status': [1, 2, 1, 2, 1]  # 1: High, 2: Low
}

personalized_data = {
    'action_id': [1, 2, 3, 4, 5],
    'environment_data': [10, 20, 30, 40, 50],
    'work_success': [0.8, 0.4, 0.9, 0.2, 1.0],
    'financial_gain': [100, 50, 150, 30, 200],
    'social_interaction': [0.5, 0.3, 0.7, 0.2, 0.9],
    'stress_level': [0.1, 0.2, 0.05, 0.25, 0.0]
}

# データフレームの作成
df_statistical = pd.DataFrame(statistical_data)
df_personalized = pd.DataFrame(personalized_data)

# データの結合
df = pd.concat([df_statistical, df_personalized], axis=1)

# 報酬の計算
weights = {
    'work_success': 0.4,
    'financial_gain': 0.3,
    'social_interaction': 0.2,
    'stress_level': -0.1  # ストレスレベルは負の影響として扱う
}

df['total_reward'] = (
    df['work_success'] * weights['work_success'] +
    df['financial_gain'] * weights['financial_gain'] +
    df['social_interaction'] * weights['social_interaction'] +
    df['stress_level'] * weights['stress_level']
)

# データの前処理
X = df[['age', 'gender', 'region', 'economic_status', 'action_id', 'environment_data']]
y = df['total_reward']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# モデルの構築と学習
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')

# モデルの保存
joblib.dump(model, 'personalized_behavior_model.pkl')

# LLMを使用した行動の評価
# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# パイプラインの設定
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# 行動データを自然言語に変換
actions = [
    "Action 1: High work success, medium financial gain, low social interaction, low stress level",
    "Action 2: Low work success, medium financial gain, medium social interaction, medium stress level",
    "Action 3: High work success, high financial gain, high social interaction, low stress level",
    "Action 4: Low work success, low financial gain, low social interaction, high stress level",
    "Action 5: High work success, very high financial gain, high social interaction, no stress level"
]

# 行動の評価
action_evaluations = nlp(actions)

# 結果の表示
for action, evaluation in zip(actions, action_evaluations):
    print(f'{action} -> {evaluation}')

# ベストな行動の選択
best_action = max(action_evaluations, key=lambda x: x['score'])
print(f'Best Action: {best_action}')

# フィードバックループの導入
new_action_result = {
    'age': 30,
    'gender': 0,  # Male
    'region': 1,  # Urban
    'economic_status': 1,  # High
    'action_id': 6,
    'environment_data': 35,
    'work_success': 0.7,
    'financial_gain': 80,
    'social_interaction': 0.6,
    'stress_level': 0.15
}

df = df.append(new_action_result, ignore_index=True)

X = df[['age', 'gender', 'region', 'economic_status', 'action_id', 'environment_data']]
y = df['total_reward']
X_scaled = scaler.fit_transform(X)

model.fit(X_scaled, y)

# モデルの保存
joblib.dump(model, 'personalized_behavior_model_updated.pkl')