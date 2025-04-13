# -*- coding: utf-8 -*-
import streamlit as st
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        max-width: 90% !important;
    }
    </style>
""", unsafe_allow_html=True)
import sqlite3
import openai
# ★★★ OpenAI v1.x 対応: OpenAI クラスをインポート ★★★
from openai import OpenAI
import requests
import json
import os
import datetime
from dotenv import load_dotenv
import os
from PIL import Image
import io
import pandas as pd
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
header_base64 = get_base64_image("assets/header_okosy.png")

st.markdown(
    f"""
    <div style="text-align: center; margin-top: 30px; margin-bottom: 100px;">
        <img src="data:image/png;base64,{header_base64}" width="700" style="border-radius: 8px;">
    </div>
    """,
    unsafe_allow_html=True
)
#CSS設定
st.markdown("""
    <style>
    .title-center {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #246798;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
/* ボタンを中央に置くラッパー */
.button-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 30px;
    margin-bottom: 60px;
}

/* Streamlitのbuttonにスタイルを当てる */
div.stButton > button {
 background-color: transparent; /* 背景は透明＝白抜き */
    color: #246798; /* テキストカラーは青 */
    border: 1.5pt solid #246798; /* 枠線も青で1.5pt */
    padding: 0.75em 2.5em;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    transition: transform 0.2s ease, background-color 0.4s ease, color 0.4s ease;
}

div.stButton > button:hover {
    background-color: #EAEAEA;     /* 背景は薄いグレー */
    color: #666666;                /* テキストはグレーに */
    border: none;                  /* 枠線を消す */
    transform: scale(1.05);        /* 少し拡大 */
}
</style>
""", unsafe_allow_html=True)

# --- 1. 環境変数の読み込みと初期設定 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
from google.cloud import vision
# --- 認証設定 ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not OPENAI_API_KEY:
    st.error("OpenAI APIキーが見つかりません。.envファイルを確認してください。")
    st.stop()

if not GOOGLE_PLACES_API_KEY:
    st.error("Google Places APIキーが見つかりません。.envファイルを確認してください。")
    st.stop()

# ★★★クライアントを初期化 ★★★
client = OpenAI()

# --- 2. データベースの初期設定 (SQLite) ---
DATABASE_NAME = "okosy_data_noauth.db"

def get_db_connection():#SQLiteデータベースへのコネクション関数を定義
    return sqlite3.connect(DATABASE_NAME)

def init_db():
    """
    データベースのテーブルが存在しなければ作成する
    - しおり(itineraries)
    - 思い出(memories)
    """
    conn = get_db_connection()#関数名を置き換え
    cursor = conn.cursor()#データベースをいじるためのカーソル定義
    # しおりテーブル (usernameなし)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS itineraries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            preferences TEXT,
            generated_content TEXT,
            places_data TEXT,
            creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')#しおりのデータベースを作成(カラム名を書き、その右に型を記載(text, not nullなど))

    # 思い出テーブル
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            itinerary_id INTEGER NOT NULL,
            caption TEXT,
            photo BLOB,
            creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (itinerary_id) REFERENCES itineraries (id)
        )
    ''')#同様に思い出テーブルのデータベースを作成
    conn.commit()
    conn.close()

# 先ほど定義したDB関数を実行(データベースがない時には作る、あればスルー)
init_db()

# --- 3. 認証関連コードをここに組み込む ---

# --- 4. Google Maps関連のヘルパー関数 ---
def get_coordinates(address):#GoogleのジオコーディングAPIを叩いて、addressに入れた情報（例えば「京都」）から経度緯度（lat/lng）を取得
    geocode_url = "https://maps.googleapis.com/maps/api/geocode/json" #APIのURL
    params = {
    "address": address,                      # 調べたい住所や地名（例："京都駅"）
    "key": GOOGLE_PLACES_API_KEY,            #Google APIキー（アクセス権）
    "language": "ja",                        # 結果を日本語にする
    "region": "JP"                           # 日本の情報を優先する
    }

    try:
        response = requests.get(geocode_url, params=params) #APIにparamで設定したパラメータを送る
        response.raise_for_status() #200以外でエラーが出るように設定
        results = response.json() #返ってきたjsonデータを辞書形式で受ける
        if results["status"] == "OK" and results["results"]: #結果が返ってきていれば
            location = results["results"][0]["geometry"]["location"] #結果の一番最初の緯度経度と位置情報を取得
            return f"{location['lat']},{location['lng']}" #必要なのは緯度経度の文字情報なので、文字情報に変換
        else:
            print(f"Geocoding失敗: {results.get('status')}, {results.get('error_message', '')}")
            return None #エラーが出たら、Noneにして処理を進める
    except Exception as e:
        print(f"Geocodingエラー: {e}")
        return None #エラーが出たら、Noneにして処理を進める、パート2
    
        
# VisionAPIを用い、画像ラベル抽出用の関数を追加
def get_vision_labels_from_uploaded_images(images):
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request

    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    if not creds.valid:
        creds.refresh(Request())

    access_token = creds.token
    endpoint = "https://vision.googleapis.com/v1/images:annotate"
    all_labels = []

    for img_file in images:
        content = base64.b64encode(img_file.read()).decode("utf-8")
        payload = {
            "requests": [{
                "image": {"content": content},
                "features": [{"type": "LABEL_DETECTION", "maxResults": 5}]
            }]
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            labels = [ann["description"] for ann in data["responses"][0].get("labelAnnotations", [])]
            all_labels.extend(labels)
        else:
            print("Vision API REST error:", response.text)

    unique_labels = list(set(all_labels))
    return unique_labels[:10]



def search_google_places(query: str,  # 検索キーワード（例: “静かなカフェ”）
                         location_bias: str = None,  # 緯度・経度の文字列（例: "35.68,139.76"）
                         place_type: str = "tourist_attraction",  # 場所の種類（初期値は「観光名所」）
                         min_rating: float = 4.0,  # 最低評価（星4以上）
                         price_levels: str = None):  # 価格帯（例: "1,2" → 安め〜普通）
    print("--- Google Places API 呼び出し ---")
    print(f"Query: {query}, Location Bias: {location_bias}, Type: {place_type}, Rating: {min_rating}, Price: {price_levels}")
    #入力条件がコンソールに見えるようにするため用(streamlitでは不要)
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"#場所をテキストで検索するようのエンドポイント
    params = {
        "query": query,
        "key": GOOGLE_PLACES_API_KEY,
        "language": "ja",
        "region": "JP",
        "type": place_type,
    }#queryがついただけでさっきと一緒
    if location_bias:
        params["location"] = location_bias
        params["radius"] = 20000
    print(f"リクエストパラメータ: {params}")#コンソールにAPIに送られる条件を表示
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        results = response.json()
        status = results.get("status")#こちらは複数条件が不要なので、Statusを確認するために定義を使用
        if status == "OK":
            filtered_places = []
            count = 0
            for place in results.get("results", []):
                place_rating = place.get("rating", 0)#ratingを取得
                place_price = place.get("price_level", None)#価格帯を取得
                if place_rating < min_rating: continue#4.0以下は弾く
                if price_levels:#price levelに入力があるなら…
                    try:
                        allowed_levels = [int(x.strip()) for x in price_levels.split(',')]#price levelを整数リストに
                        if place_price not in allowed_levels: continue #ユーザーの価格帯に合わなければスキップ
                    except ValueError: print(f"価格レベルの解析エラー: {price_levels}") #price levelのエラー時に取り除く処理
                filtered_places.append({
                    "name": place.get("name"), "address": place.get("formatted_address"),
                    "rating": place_rating, "price_level": place_price,
                    "types": place.get("types", []), "place_id": place.get("place_id"),
                })#filtered_placesの辞書に必要な情報をどんどん追加していっている
                count += 1
                if count >= 5: break #5つ目までで打ち止め
            if not filtered_places:#filter placeが空なら
                return json.dumps({"error": "条件に合致する場所がありませんでした。"}, ensure_ascii=False)#エラーを返す
            return json.dumps(filtered_places, ensure_ascii=False)# 作った filtered_places リストをJSON形式で返す
        else:#APIエラーの場合、それを返す
            error_msg = results.get('error_message', '')
            print(f"Google Places API エラー: {status}, {error_msg}")
            return json.dumps({"error": f"Google Places API Error: {status}, {error_msg}"}, ensure_ascii=False)
    except Exception as e:#そもそものtryに対するエラー
        print(f"HTTPリクエストエラー: {e}")
        return json.dumps({"error": f"HTTPエラー: {e}"}, ensure_ascii=False)


# --- 5. OpenAIのFunction Callingを組み込むための準備 ---

# ★★★ OpenAI v1.x 対応: functions -> tools 形式に変更 ★★★
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_google_places",
            "description": "Google Places APIを使って観光名所やレストランなどを検索する。隠れ家的なお店や、静かなカフェ、旅館など具体的な場所情報が必要なときに呼び出す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索したい場所のキーワード (例: '京都 抹茶 スイーツ')"},
                    "location_bias": {"type": "string", "description": "検索の中心とする緯度経度 (例: '35.0116,135.7681')。行き先の座標を指定すると精度が上がる。"},
                    "place_type": {
                        "type": "string",
                        "description": "検索する場所の種類",
                        "enum": [
                            "tourist_attraction", "restaurant", "lodging", "cafe",
                            "museum", "park", "art_gallery", "store"
                        ]
                    },
                    "min_rating": {"type": "number", "description": "検索結果に含める最低評価 (例: 4.0)"},
                    "price_levels": {"type": "string", "description": "検索結果に含める価格帯（カンマ区切り、例: '1,2'）。1:安い, 2:普通, 3:やや高い, 4:高い"}
                },
                "required": ["query", "place_type"]
            }
        }
    }
]
# --- ここまで tools 定義 ---

# --- 呼び出し可能な関数名と実際の関数オブジェクトのマッピング (変更なし) ---
available_functions = {
    "search_google_places": search_google_places
}#先ほどdefで定義した関数と、上記のtoolsのFunctionの名前を紐づけている
# --- ここまで 関数マッピング ---

# ★★★ OpenAI v1.x 対応: API呼び出しとレスポンス処理を修正 ★★★
def run_conversation_with_function_calling(messages):
    try:
        # 1回目: GPTに会話を送信（tool callを期待）
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # Tool callメッセージを履歴に追加
            messages.append(response_message)

            # すべてのtool_callに対応
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                function_args = json.loads(tool_call.function.arguments)

                # location_bias の補完
                if 'location_bias' not in function_args and 'dest' in st.session_state:
                    coords = get_coordinates(st.session_state.dest)
                    if coords:
                        function_args['location_bias'] = coords

                # 関数を実行
                function_response = function_to_call(**function_args)

                # tool role のレスポンスを履歴に追加
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })

            # 2回目: ツールの結果を含めて再送信
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            print("=== 2回目のGPT応答 ===")
            print(second_response.choices[0].message)
            
            final_content = second_response.choices[0].message.content
            return final_content, function_response

        else:
            # tool_callがない場合は普通に返す
            final_content = response_message.content
            return final_content, None

    except openai.APIError as e:
        st.error(f"OpenAI APIエラーが発生しました: {e}")
        return "APIエラーが発生しました。", None
    except Exception as e:
        st.error(f"OpenAIとの通信中に予期せぬエラーが発生しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        return "エラーが発生しました。", None


# --- 6. Streamlitの画面構成 (認証なし) ---


# --- サイドバー ---
with st.sidebar:
    st.image("assets/logo_okosy.png", width=100)
st.sidebar.header("メニュー")
menu_choice = st.sidebar.radio("", ["新しい旅を計画する", "過去の旅のしおりを見る"], key="main_menu", label_visibility="collapsed")

# --- セッションステート初期化(この後の定義文字と合うように記載している) ---
if "show_planner_select" not in st.session_state:
    st.session_state.show_planner_select = False
if "planner_selected" not in st.session_state:
    st.session_state.planner_selected = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "itinerary_generated" not in st.session_state:
    st.session_state.itinerary_generated = False
# ...(他のセッションステートも同様)...
if "generated_shiori_content" not in st.session_state:
    st.session_state.generated_shiori_content = None
if "final_places_data" not in st.session_state:
    st.session_state.final_places_data = None
if "basic_info_submitted" not in st.session_state:
    st.session_state.basic_info_submitted = False
if "preferences_submitted" not in st.session_state:
    st.session_state.preferences_submitted = False
if "preferences" not in st.session_state:
    st.session_state.preferences = {}


# --- メインコンテンツ ---

# --- 7. 新しい旅を計画する ---
from PIL import Image
st.markdown("""
    <style>
    .centered-button {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-center">さあ、あなただけの旅をはじめよう。</div>', unsafe_allow_html=True)

st.markdown('<div class="center-button-wrapper">', unsafe_allow_html=True)
start_clicked = st.button("プランニングを始める")
st.markdown('</div>', unsafe_allow_html=True)

if start_clicked:
    st.session_state.show_planner_select = True

if st.session_state.show_planner_select and not st.session_state.planner_selected:
    st.subheader("あなたにぴったりのプランナーを選んでください")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("シゴデキのベテランプランナー"):
            st.session_state.planner = "ベテラン"
        st.caption("テイスト：端的でシンプル。安心のプロ感。")

        if st.button("地元に詳しいおせっかい姉さん"):
            st.session_state.planner = "姉さん"
        st.caption("テイスト：その土地の方言＋親しみやすさ満点。")

    with col2:
        if st.button("旅好きインスタグラマー"):
            st.session_state.planner = "ギャル"
        st.caption("テイスト：テンション高め、語尾にハート。")

        if st.button("甘い言葉をささやく王子様"):
            st.session_state.planner = "王子"
        st.caption("テイスト：ちょっとナルシストだけど優しくリード。")
    
if menu_choice == "新しい旅を計画する":
    st.header("新しい旅の計画")
    st.subheader("1. 旅の基本情報を入力")

    with st.form("basic_info_form"):
        # フォーム要素 (変更なし)
        st.session_state.dest = st.text_input("行き先 (例: 京都、箱根)", value=st.session_state.get('dest', ''))
        st.session_state.purp = st.text_area("旅の目的や気分", value=st.session_state.get('purp', ''))
        st.session_state.comp = st.selectbox("同行者", ["一人旅", "夫婦・カップル", "友人", "家族"], index=["一人旅", "夫婦・カップル", "友人", "家族"].index(st.session_state.get('comp', '一人旅')))
        st.session_state.days = st.number_input("旅行日数", min_value=1, max_value=30, step=1, value=st.session_state.get('days', 2))
        st.session_state.budg = st.select_slider("予算感", options=["気にしない", "安め", "普通", "高め"], value=st.session_state.get('budg', "普通"))
        submitted_basic = st.form_submit_button("基本情報を確定")
#select box→選択肢の中から選ぶ
#select_slider→評価をスライダー形式で選ぶ
#value = は、次のセッションで使う用に結果を保存している

    if submitted_basic:
        if not st.session_state.dest:
            st.warning("行き先を入力してください。")
        else:
            st.success(f"基本情報を受け付けました: {st.session_state.dest}への{st.session_state.comp}旅行 ({st.session_state.days}日間)")
            st.session_state.basic_info_submitted = True
            st.session_state.itinerary_generated = False
            st.session_state.generated_shiori_content = None
            st.session_state.final_places_data = None
            st.session_state.preferences_submitted = False
            st.session_state.preferences = {}

    if st.session_state.basic_info_submitted:
        st.subheader("2. あなたの好みを教えてください")

        st.subheader("画像からインスピレーションを得る")
        uploaded_images = st.file_uploader("あなたが「好き」と思う写真を3枚までアップロードしてください（自然、街並み、空間など）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_images:
            st.session_state.uploaded_images = uploaded_images[:3]

        with st.form("preferences_form"):
            st.session_state.pref_pace = st.radio("旅のペースは？", ["のんびり", "普通", "アクティブ"], index=["のんびり", "普通", "アクティブ"].index(st.session_state.get('pref_pace', '普通')))
            st.session_state.pref_nature = st.slider("自然(1～5)", 1, 5, st.session_state.get('pref_nature', 3))
            st.session_state.pref_culture = st.slider("歴史文化(1～5)", 1, 5, st.session_state.get('pref_culture', 3))
            st.session_state.pref_art = st.slider("アート(1～5)", 1, 5, st.session_state.get('pref_art', 3))
            st.session_state.pref_food_local = st.radio("食事スタイル", ["地元の人気店", "隠れ家的なお店", "こだわらない"], index=["地元の人気店", "隠れ家的なお店", "こだわらない"].index(st.session_state.get('pref_food_local', '地元の人気店')))
            st.session_state.pref_food_style = st.multiselect("好きな料理", ["和食", "洋食", "カフェ", "スイーツ", "居酒屋"], default=st.session_state.get('pref_food_style', []))
            st.session_state.pref_accom_type = st.radio("宿タイプ", ["ホテル", "旅館", "民宿・ゲストハウス", "こだわらない"], index=["ホテル", "旅館", "民宿・ゲストハウス", "こだわらない"].index(st.session_state.get('pref_accom_type', 'ホテル')))
            st.session_state.pref_accom_view = st.checkbox("宿の景色重視", value=st.session_state.get('pref_accom_view', False))
            st.session_state.pref_vibe_quiet = st.radio("好み雰囲気", ["静かで落ち着いた", "活気のある"], index=["静かで落ち着いた", "活気のある"].index(st.session_state.get('pref_vibe_quiet', '静かで落ち着いた')))
            st.session_state.pref_vibe_discover = st.checkbox("隠れた発見をしたい", value=st.session_state.get('pref_vibe_discover', True))
            st.session_state.pref_experience = st.multiselect("興味ある体験", ["温泉", "ものづくり", "寺社仏閣", "食べ歩き", "ショッピング", "何もしない"], default=st.session_state.get('pref_experience', []))
            submitted_prefs = st.form_submit_button("好みを確定して旅のしおりを生成")

            if submitted_prefs:
                st.session_state.preferences_submitted = True
                st.session_state.preferences = {
                    "pace": st.session_state.pref_pace, "nature": st.session_state.pref_nature,
                    "culture": st.session_state.pref_culture, "art": st.session_state.pref_art,
                    "food_local": st.session_state.pref_food_local, "food_style": st.session_state.pref_food_style,
                    "accom_type": st.session_state.pref_accom_type, "accom_view": st.session_state.pref_accom_view,
                    "vibe_quiet": st.session_state.pref_vibe_quiet, "vibe_discover": st.session_state.pref_vibe_discover,
                    "experience": st.session_state.pref_experience
                }
                st.info("しおりを作成中です。少々お待ちください...")

                vision_tags = []
                if "uploaded_images" in st.session_state and st.session_state.uploaded_images:
                    try:
                        st.info("画像からあなたの好みを解析しています...")
                        vision_tags = get_vision_labels_from_uploaded_images(st.session_state.uploaded_images)
                    except Exception as e:
                        st.warning(f"画像解析中にエラーが発生しました: {e}")
                        vision_tags = []
                vision_tag_text = "、".join(vision_tags) if vision_tags else "（画像からの情報はありません）"

                prompt = f"""
あなたは旅のプランナー「Okosy」です。ユーザーの入力情報をもとに、SNS映えや定番から少し離れた、ユーザー自身の感性に寄り添うような、パーソナルな旅のしおりを作成してください。

【画像から読み取れた特徴（Google Vision APIによるラベル抽出）】
{vision_tag_text}

【基本情報】
- 行き先: {st.session_state.dest}
- 目的・気分: {st.session_state.purp}
- 同行者: {st.session_state.comp}
- 旅行日数: {st.session_state.days}日
- 予算感: {st.session_state.budg}

【ユーザーの好み】
{json.dumps(st.session_state.preferences, ensure_ascii=False, indent=2)}

【出力指示】
1. 各日を「午前」「午後」「夜」に分けて提案してください。
2. なぜその場所が合っているのか、感性的な理由を含めてください。
3. 必要に応じて search_google_places を使用してください。
4. 出力形式はマークダウンです。
            """

                st.session_state.messages = [{"role": "user", "content": prompt}]
                with st.spinner("AIが旅のしおりを作成しています..."):
                    final_response, places_api_result = run_conversation_with_function_calling(st.session_state.messages)

                if final_response:
                    st.session_state.itinerary_generated = True
                    st.session_state.generated_shiori_content = final_response
                    st.session_state.final_places_data = places_api_result
                    st.success("旅のしおりが完成しました！")

                    st.subheader("あなたの旅のしおり")
                    st.markdown(st.session_state.generated_shiori_content)
                else:
                    st.error("しおりの生成中にエラーが発生しました。")