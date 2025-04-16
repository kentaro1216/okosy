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

# --- 1. 必要なライブラリのインポート ---
import openai
from openai import OpenAI
import requests
import json
import os
import datetime
from dotenv import load_dotenv
from PIL import Image
import io
import pandas as pd
import random
import time
import base64
import traceback
from typing import Optional, List, Dict, Any # <<< 型ヒント追加

# Firebase 関連ライブラリ
import firebase_admin
from firebase_admin import credentials, auth, firestore
# Streamlit Firebase Auth コンポーネント
try:
    import streamlit_firebase_auth as sfa
except ImportError:
    st.error("認証ライブラリが見つかりません。`pip install streamlit-firebase-auth` を実行してください。")
    sfa = None
except Exception as e:
    st.error(f"streamlit-firebase-auth のインポート中に予期せぬエラー: {e}")
    sfa = None

# Google Cloud Vision
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
except ImportError:
    st.error("Google Cloud Visionライブラリが見つかりません。`pip install google-cloud-vision google-auth` を実行してください。")
    vision = None # type: ignore
    service_account = None # type: ignore
    Request = None # type: ignore

# --- ヘッダー画像表示 ---
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"ヘッダー画像ファイルが見つかりません: {image_path}")
        return None
    except Exception as e:
        st.warning(f"ヘッダー画像の読み込み中にエラー: {e}")
        return None

header_base64 = get_base64_image("assets/header_okosy.png")
if header_base64:
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 30px; margin-bottom: 100px;">
            <img src="data:image/png;base64,{header_base64}" width="700" style="border-radius: 8px;">
        </div>
        """,
        unsafe_allow_html=True
    )

# --- CSS設定 ---
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
        background-color: #EAEAEA;      /* 背景は薄いグレー */
        color: #666666;                 /* テキストはグレーに */
        border: none;                   /* 枠線を消す */
        transform: scale(1.05);         /* 少し拡大 */
    }
    /* プランナー選択ボタンのスタイル */
    .planner-button > button {
        width: 100%; /* カラム幅いっぱいに広げる */
        margin-bottom: 10px; /* ボタン間の余白 */
    }
    /* 中央揃えのボタンラッパー */
    .center-button-wrapper { /* このクラス名を button-wrapper から変更 */
        display: flex;
        justify-content: center;
        margin-top: 30px; /* 上の余白 */
        margin-bottom: 60px; /* 下の余白 */
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. 環境変数の読み込みと初期設定 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
SERVICE_ACCOUNT_KEY_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_PATH", "path/to/your/serviceAccountKey.json")
FIREBASE_CONFIG_PATH = os.getenv("FIREBASE_CONFIG_PATH", "path/to/your/firebase_config.json")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# --- 認証設定 (Vision API用) ---
if GOOGLE_APPLICATION_CREDENTIALS:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
else:
    st.warning("環境変数 GOOGLE_APPLICATION_CREDENTIALS が設定されていません。Vision APIが利用できない可能性があります。")

# APIキーの存在チェック
if not OPENAI_API_KEY:
    st.error("OpenAI APIキーが見つかりません。.envファイルを確認してください。")
    st.stop()

if not GOOGLE_PLACES_API_KEY:
    st.error("Google Places APIキーが見つかりません。.envファイルを確認してください。")
    st.stop()

# --- OpenAI クライアント初期化 ---
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"OpenAIクライアントの初期化に失敗しました: {e}")
    st.stop()

# --- 2. Firebase Admin SDK の初期化 ---
if not firebase_admin._apps:
    if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        st.error(f"Firebase サービスアカウントキーが見つかりません: {SERVICE_ACCOUNT_KEY_PATH}\n.envファイルで FIREBASE_SERVICE_ACCOUNT_KEY_PATH を設定するか、パスを直接指定してください。")
        st.stop()
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        st.error(f"Firebase Admin SDK の初期化に失敗しました: {e}")
        st.error(traceback.format_exc())
        st.stop()

# --- 2.1 Firebase Web アプリ設定の読み込み ---
firebase_config = None
if not os.path.exists(FIREBASE_CONFIG_PATH):
    st.error(f"Firebase Web 設定ファイルが見つかりません: {FIREBASE_CONFIG_PATH}\n.envファイルで FIREBASE_CONFIG_PATH を設定するか、パスを直接指定してください。")
    st.stop()
else:
    try:
        with open(FIREBASE_CONFIG_PATH) as f:
            firebase_config = json.load(f)
    except Exception as e:
        st.error(f"Firebase Web 設定ファイルの読み込みに失敗しました: {e}")
        st.stop()

# --- 2.2 streamlit-firebase-auth コンポーネントの初期化 ---
auth_obj = None
if sfa is None:
    st.error("認証機能の初期化に失敗しました。streamlit-firebase-authが正しくインストールされているか確認してください。")
    st.stop()
if firebase_config is None:
    st.error("Firebase Web設定が読み込めていないため、認証機能を初期化できません。")
    st.stop()

try:
    auth_obj = sfa.FirebaseAuth(firebase_config)
except Exception as e:
    st.error(f"FirebaseAuth オブジェクトの作成に失敗しました: {e}")
    st.error(traceback.format_exc())
    st.stop()

# --- 3.3 Firestore クライアントの初期化 ---
try:
    db = firestore.client()
    print("Firestore client initialized successfully.")
except Exception as e:
    st.error(f"Firestore クライアントの初期化に失敗しました: {e}")
    st.error(traceback.format_exc())
    st.stop()

# --- Firestore データ操作関数 --- (変更なし)
def save_itinerary_to_firestore(user_id: str, name: str, preferences: dict, generated_content: str, places_data: Optional[str]):
    """しおりデータをFirestoreに保存する"""
    if not db:
        st.error("Firestoreクライアントが初期化されていません。しおりを保存できません。")
        return None
    try:
        doc_ref = db.collection("users").document(user_id).collection("itineraries").document()
        doc_ref.set({
            "name": name,
            "preferences": json.dumps(preferences, ensure_ascii=False),
            "generated_content": generated_content,
            "places_data": places_data if places_data else None,
            "creation_date": firestore.SERVER_TIMESTAMP # type: ignore
        })
        print(f"Itinerary saved to Firestore for user {user_id}, doc_id: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        st.error(f"Firestoreへのしおり保存中にエラー: {e}")
        print(traceback.format_exc())
        return None

def load_itineraries_from_firestore(user_id: str):
    """指定したユーザーのしおり一覧をFirestoreから読み込む"""
    if not db: return []
    itineraries = []
    try:
        itineraries_ref = db.collection("users").document(user_id).collection("itineraries").order_by(
            "creation_date", direction=firestore.Query.DESCENDING # type: ignore
        ).stream()
        for doc in itineraries_ref:
            data = doc.to_dict()
            if data:
                data['id'] = doc.id
                try:
                    # JSON文字列から辞書に変換
                    data['preferences_dict'] = json.loads(data.get('preferences', '{}'))
                except (json.JSONDecodeError, TypeError):
                    data['preferences_dict'] = {} # エラー時は空の辞書
                itineraries.append(data)
        return itineraries
    except Exception as e:
        st.error(f"Firestoreからのしおり読み込み中にエラー: {e}")
        print(traceback.format_exc())
        return []

def delete_itinerary_from_firestore(user_id: str, itinerary_id: str):
    """指定したしおりと関連する思い出をFirestoreから削除する"""
    if not db: return False
    try:
        # まずサブコレクション(memories)を削除 (バッチ処理)
        memories_ref = db.collection("users").document(user_id).collection("itineraries").document(itinerary_id).collection("memories").stream()
        batch_mem = db.batch()
        mem_deleted_count = 0
        for mem_doc in memories_ref:
            batch_mem.delete(mem_doc.reference)
            mem_deleted_count += 1
        if mem_deleted_count > 0:
            batch_mem.commit()
            print(f"Deleted {mem_deleted_count} memories for itinerary {itinerary_id}")

        # 次にしおり本体を削除
        db.collection("users").document(user_id).collection("itineraries").document(itinerary_id).delete()

        print(f"Itinerary {itinerary_id} deleted from Firestore for user {user_id}")
        return True
    except Exception as e:
        st.error(f"Firestoreからのしおり削除中にエラー: {e}")
        print(traceback.format_exc())
        return False

def save_memory_to_firestore(user_id: str, itinerary_id: str, caption: str, photo_base64: Optional[str]):
    """思い出データをFirestoreに保存する (写真はBase64文字列)"""
    if not db: return None
    try:
        doc_ref = db.collection("users").document(user_id).collection("itineraries").document(itinerary_id).collection("memories").document()
        doc_ref.set({
            "caption": caption,
            "photo_base64": photo_base64, # Noneの場合もそのまま保存
            "creation_date": firestore.SERVER_TIMESTAMP # type: ignore
        })
        print(f"Memory saved to Firestore for itinerary {itinerary_id}, doc_id: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        st.error(f"Firestoreへの思い出保存中にエラー: {e}")
        print(traceback.format_exc())
        return None

def load_memories_from_firestore(user_id: str, itinerary_id: str):
    """指定したしおりの思い出一覧をFirestoreから読み込む"""
    if not db: return []
    memories = []
    try:
        memories_ref = db.collection("users").document(user_id).collection("itineraries").document(itinerary_id).collection("memories").order_by(
            "creation_date", direction=firestore.Query.DESCENDING # type: ignore
        ).stream()
        for doc in memories_ref:
            data = doc.to_dict()
            if data:
                data['id'] = doc.id
                photo_b64 = data.get('photo_base64')
                if photo_b64:
                    try:
                        # Base64からデコードしてPIL Imageオブジェクトに変換
                        img_bytes = base64.b64decode(photo_b64)
                        data['photo_image'] = Image.open(io.BytesIO(img_bytes))
                    except Exception as img_e:
                        print(f"Error decoding/loading image from base64 for memory {doc.id}: {img_e}")
                        data['photo_image'] = None # エラー時はNone
                else:
                    data['photo_image'] = None # Base64データがない場合もNone
                memories.append(data)
        return memories
    except Exception as e:
        st.error(f"Firestoreからの思い出読み込み中にエラー: {e}")
        print(traceback.format_exc())
        return []

def delete_memory_from_firestore(user_id: str, itinerary_id: str, memory_id: str):
    """指定した思い出をFirestoreから削除する"""
    if not db: return False
    try:
        db.collection("users").document(user_id).collection("itineraries").document(itinerary_id).collection("memories").document(memory_id).delete()
        print(f"Memory {memory_id} deleted from Firestore for itinerary {itinerary_id}")
        return True
    except Exception as e:
        st.error(f"Firestoreからの思い出削除中にエラー: {e}")
        print(traceback.format_exc())
        return False

# --- 3. 認証処理とログイン状態の管理 --- (変更なし)
if 'user_info' not in st.session_state:
    st.session_state['user_info'] = None
if 'id_token' not in st.session_state:
    st.session_state['id_token'] = None

if st.session_state['user_info'] is None:
    st.subheader("Googleアカウントでログイン")
    st.write("Okosy を利用するには、Googleアカウントでのログインが必要です。")
    st.info("下のフォームの「Sign in with Google」ボタンをクリックしてください。\n（メールアドレス/パスワード欄は使用しません）")

    if auth_obj is None:
        st.error("認証オブジェクトが初期化されていません。")
        st.stop()

    try:
        # ログインフォームを表示
        login_result = auth_obj.login_form()
        # login_result の構造を慎重にチェック
        if login_result and isinstance(login_result, dict) and login_result.get('success') is True:
            user_data = login_result.get('user')
            if user_data and isinstance(user_data, dict):
                token_manager = user_data.get('stsTokenManager')
                if token_manager and isinstance(token_manager, dict):
                    id_token = token_manager.get('accessToken')
                    if id_token:
                        st.session_state['id_token'] = id_token
                        try:
                            # トークンを検証
                            decoded_token = auth.verify_id_token(st.session_state['id_token'])
                            st.session_state['user_info'] = decoded_token
                            st.success("ログインしました！")
                            print(f"User logged in: {decoded_token.get('uid')}")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"ログイン中にエラーが発生しました (トークン検証失敗): {e}")
                            print(f"Token verification failed for token starting with {str(id_token)[:10]}... Error: {e}")
                            st.session_state['id_token'] = None
                            st.session_state['user_info'] = None
                    else:
                        st.error("ログイン成功しましたが、認証トークン(accessToken)が見つかりませんでした。")
                        print("Login success reported, but accessToken not found in stsTokenManager.")
                else:
                    st.error("ログイン成功しましたが、トークン情報(stsTokenManager)が見つかりませんでした。")
                    print("Login success reported, but stsTokenManager not found in user data.")
            else:
                st.error("ログイン成功しましたが、ユーザー情報(user)が見つかりませんでした。")
                print("Login success reported, but user data not found in result.")
        elif login_result and isinstance(login_result, dict) and login_result.get('success') is False:
            error_message = login_result.get('error', '不明なエラー')
            # よくあるエラー（ポップアップブロックなど）に対するユーザーフレンドリーなメッセージ
            if 'auth/popup-closed-by-user' in str(error_message):
                 st.warning("ログインポップアップが閉じられたか、ブロックされたようです。ポップアップを許可して再試行してください。")
            elif 'auth/cancelled-popup-request' in str(error_message):
                 st.warning("ログインリクエストがキャンセルされました。")
            else:
                 st.error(f"ログインに失敗しました: {error_message}")
            print(f"Login failed: {error_message}")

    except Exception as e:
        st.error(f"認証フォームの表示または処理中にエラーが発生しました: {e}")
        st.error(traceback.format_exc())
    st.stop()

# --- 3.1 ログイン後のメインコンテンツ ---
if st.session_state.get('user_info') is not None:
    user_id = st.session_state['user_info'].get('uid') # <<< ユーザーIDを取得
    if not user_id:
        st.error("ユーザーIDが取得できませんでした。再ログインしてください。")
        st.session_state['user_info'] = None
        st.session_state['id_token'] = None
        st.rerun()

    # --- サイドバーの設定 (ログイン後) ---
    st.sidebar.header("メニュー")
    user_email = st.session_state['user_info'].get('email', '不明なユーザー')
    st.sidebar.write(f"ログイン中: {user_email}")

    # ログアウトボタン
    if st.sidebar.button("ログアウト"):
        st.session_state['user_info'] = None
        st.session_state['id_token'] = None
        # ログアウト時にクリアするセッションステートキーのリスト
        keys_to_clear_on_logout = [
            "itinerary_generated", "generated_shiori_content", "final_places_data",
            "preferences_for_prompt", "determined_destination", "determined_destination_for_prompt",
            "messages_for_prompt", "shiori_name_input", "selected_itinerary_id", "selected_itinerary_id_selector",
            "show_planner_select", "planner_selected", "planner",
            "messages", "basic_info_submitted", "preferences_submitted", "preferences",
            "dest", "purp", "comp", "days", "budg", "pref_nature", "pref_culture", "pref_art", "pref_welness",
            "pref_food_local", "pref_food_style", "pref_accom_type", "pref_word", "mbti",
            "pref_food_style_ms", "pref_word_ms", "mbti_input", # フォーム入力用のキーもクリア
            "uploaded_image_files", "q0_answer", "q1_answer", "q2_answer", # <<< uploaded_images -> uploaded_image_files, qX_answer もクリア対象に追加
            "memory_caption", "memory_photo" # 思い出フォームのキーもクリア
        ]
        # 存在する場合のみ削除
        for key in keys_to_clear_on_logout:
            if key in st.session_state:
                del st.session_state[key]
        st.success("ログアウトしました。")
        print("User logged out.")
        time.sleep(1)
        st.rerun()

    st.sidebar.markdown("---")
    # サイドバーメニュー選択
    menu_choice = st.sidebar.radio("", ["新しい旅を計画する", "過去の旅のしおりを見る"], key="main_menu", label_visibility="collapsed")
    st.sidebar.image("assets/logo_okosy.png", width=100)

    # --- 4. Google Maps関連のヘルパー関数 ---
    def get_coordinates(address):
        """Google Geocoding APIを使用して住所から緯度経度を取得する"""
        geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": address,
            "key": GOOGLE_PLACES_API_KEY,
            "language": "ja",
            "region": "JP"
        }
        try:
            response = requests.get(geocode_url, params=params, timeout=10) # タイムアウト追加
            response.raise_for_status()
            results = response.json()
            if results["status"] == "OK" and results["results"]:
                location = results["results"][0]["geometry"]["location"]
                return f"{location['lat']},{location['lng']}"
            else:
                print(f"Geocoding failed: Status={results.get('status')}, Error={results.get('error_message', '')}")
                return None
        except requests.exceptions.Timeout:
            print(f"Geocoding timeout for address: {address}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Geocoding HTTP error: {e}")
            return None
        except Exception as e:
            print(f"Geocoding unexpected error: {e}")
            return None

    # --- Vision API ラベル抽出関数 ---
    def get_vision_labels_from_uploaded_images(image_files):
        """アップロードされた画像ファイルからVision APIでラベルを抽出"""
        if not vision or not service_account or not Request or not GOOGLE_APPLICATION_CREDENTIALS:
             st.warning("Vision APIの利用に必要なライブラリまたは認証情報が不足しています。")
             return []
        try:
            # サービスアカウント認証情報を使用して認証
            creds = service_account.Credentials.from_service_account_file(
                GOOGLE_APPLICATION_CREDENTIALS,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            # トークンが有効か確認し、必要ならリフレッシュ
            if not creds.valid:
                creds.refresh(Request())

            access_token = creds.token
            endpoint = "https://vision.googleapis.com/v1/images:annotate"
            all_labels = []
            processed_count = 0

            for img_file in image_files:
                try:
                    # ファイルポインタをリセット
                    if hasattr(img_file, 'seek'):
                        img_file.seek(0)
                    # 画像コンテンツをBase64エンコード
                    content = base64.b64encode(img_file.read()).decode("utf-8")
                    payload = {
                        "requests": [{
                            "image": {"content": content},
                            "features": [{"type": "LABEL_DETECTION", "maxResults": 5}] # 上位5件のラベルを取得
                        }]
                    }
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    }
                    # Vision APIにリクエスト送信 (タイムアウト設定)
                    response = requests.post(endpoint, headers=headers, json=payload, timeout=20)

                    if response.status_code == 200:
                        data = response.json()
                        # レスポンスからラベル情報を抽出
                        if data.get("responses") and data["responses"][0]:
                            labels = [ann["description"] for ann in data["responses"][0].get("labelAnnotations", [])]
                            all_labels.extend(labels)
                            processed_count += 1
                        else:
                            print(f"Vision API: Empty or invalid response for one image: {data}")
                    else:
                        print(f"Vision API REST error: {response.status_code}, {response.text}")
                except requests.exceptions.Timeout:
                    st.warning(f"画像の一つでVision APIへのリクエストがタイムアウトしました。")
                    print(f"Vision API request timeout for one image.")
                    continue # 次の画像の処理へ
                except Exception as img_e:
                    st.warning(f"個別の画像処理中にエラーが発生しました: {img_e}")
                    print(f"Error processing individual image with Vision API: {img_e}")
                    continue # 次の画像の処理へ

            # 重複を除去して上位10件までを返す
            unique_labels = list(set(all_labels))
            print(f"Vision API processed {processed_count}/{len(image_files)} images. Found labels: {unique_labels[:10]}")
            return unique_labels[:10]

        except Exception as e:
            st.error(f"Vision APIによるラベル抽出全体でエラーが発生しました: {e}")
            print(f"Overall error during Vision API label extraction: {e}")
            print(traceback.format_exc())
            return []

    # --- Google Places API 検索関数 ---
    def search_google_places(query: str,
                             location_bias: Optional[str] = None,
                             place_type: str = "tourist_attraction",
                             min_rating: Optional[float] = 4.0, # <<< Optionalに変更
                             price_levels: Optional[str] = None) -> str:
        """Google Places API (Text Search) を使用して場所を検索し、結果をJSON文字列で返す"""
        print("--- Calling Google Places API ---")
        print(f"Query: {query}, Location Bias: {location_bias}, Type: {place_type}, Rating: {min_rating}, Price: {price_levels}")

        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": query,
            "key": GOOGLE_PLACES_API_KEY,
            "language": "ja",
            "region": "JP",
            "type": place_type,
        }
        if location_bias:
            params["location"] = location_bias
            params["radius"] = 20000 # 20km圏内をバイアス

        print(f"Request Parameters: {params}")

        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            results = response.json()
            status = results.get("status")

            if status == "OK":
                filtered_places = []
                count = 0
                for place in results.get("results", []):
                    place_rating = place.get("rating", 0)
                    place_price = place.get("price_level")

                    # 評価フィルタ (min_rating が指定されている場合のみ適用)
                    if min_rating is not None and place_rating < min_rating: # <<< None チェック追加
                        continue

                    # 価格帯フィルタ
                    if price_levels:
                        try:
                            allowed_levels = [int(x.strip()) for x in price_levels.split(',') if x.strip().isdigit()]
                            if place_price is not None and place_price not in allowed_levels:
                                continue
                        except ValueError:
                            print(f"Invalid price_levels format: {price_levels}")

                    filtered_places.append({
                        "name": place.get("name"), "address": place.get("formatted_address"),
                        "rating": place_rating, "price_level": place_price,
                        "types": place.get("types", []), "place_id": place.get("place_id"),
                    })
                    count += 1
                    if count >= 5: break

                if not filtered_places:
                    print("No places found matching the criteria.")
                    return json.dumps({"message": "条件に合致する場所が見つかりませんでした。"}, ensure_ascii=False)
                else:
                    print(f"Found {len(filtered_places)} places.")
                    return json.dumps(filtered_places, ensure_ascii=False)

            elif status == "ZERO_RESULTS":
                 print("Google Places API returned ZERO_RESULTS.")
                 return json.dumps({"message": "検索条件に合致する場所が見つかりませんでした。"}, ensure_ascii=False)
            else:
                error_msg = results.get('error_message', '')
                print(f"Google Places API error: Status={status}, Message={error_msg}")
                return json.dumps({"error": f"Google Places API Error: {status}, {error_msg}"}, ensure_ascii=False)

        except requests.exceptions.Timeout:
             print(f"Google Places API request timeout for query: {query}")
             return json.dumps({"error": "Google Places APIへのリクエストがタイムアウトしました。"}, ensure_ascii=False)
        except requests.exceptions.RequestException as e:
            print(f"Google Places API HTTP request error: {e}")
            return json.dumps({"error": f"Google Places APIへの接続中にHTTPエラーが発生しました: {e}"}, ensure_ascii=False)
        except Exception as e:
            print(f"Unexpected error during Google Places search: {e}")
            print(traceback.format_exc())
            return json.dumps({"error": f"場所検索中に予期せぬエラーが発生しました: {e}"}, ensure_ascii=False)

    # --- 5. OpenAI Function Calling (Tool Calling) 準備 ---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_google_places",
                "description": "Google Places APIを使って観光名所、レストラン、宿泊施設などを検索します。特定の場所（例: 静かなカフェ、評価の高い旅館）の情報が必要な場合に使用してください。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "検索キーワード (例: '京都 抹茶 スイーツ', '箱根 温泉旅館 露天風呂付き')"},
                        "location_bias": {"type": "string", "description": "検索の中心とする緯度経度 (例: '35.0116,135.7681')。目的地の座標を指定すると精度が向上します。"},
                        "place_type": {
                            "type": "string",
                            "description": "検索する場所の種類。適切なものを選択してください。",
                            "enum": [
                                "tourist_attraction", "restaurant", "lodging", "cafe",
                                "museum", "park", "art_gallery", "store", "bar", "spa"
                            ]
                        },
                        "min_rating": {"type": "number", "description": "結果に含める最低評価 (例: 4.0)。指定しない場合は評価でフィルタリングしません。"},
                        "price_levels": {"type": "string", "description": "結果に含める価格帯。カンマ区切りで指定します (例: '1,2')。1:安い, 2:普通, 3:やや高い, 4:高い。"}
                    },
                    "required": ["query", "place_type"]
                }
            }
        }
    ]
    available_functions = {
        "search_google_places": search_google_places
    }

    # --- OpenAI API 会話実行関数 (Vision API連携版) ---
    def run_conversation_with_function_calling(messages: List[Dict[str, Any]],
                                               uploaded_image_files: Optional[List[Any]] = None) -> tuple[Optional[str], Optional[str]]: # <<< 型ヒント修正
        """
        OpenAIにメッセージを送信し、Tool Callがあれば実行して結果を返し、最終的な応答を得る。
        画像がアップロードされた場合、Vision APIでラベルを抽出し、テキストとしてプロンプトに追加する。
        """
        try:
            # --- Vision APIによる画像ラベル抽出 & プロンプトへの追加 ---
            if uploaded_image_files:
                print(f"--- Processing {len(uploaded_image_files)} images with Vision API ---")
                try:
                    image_labels = get_vision_labels_from_uploaded_images(uploaded_image_files)
                    if image_labels:
                        label_text = "【画像から読み取れた特徴（参考）】\n" + ", ".join(image_labels)
                        print(f"--- Vision API Labels: {label_text} ---")
                        # 最後のメッセージ(ユーザープロンプト)にラベル情報を追記
                        last_message = messages[-1]
                        # content が文字列の場合、リストに変換して追記
                        if isinstance(last_message.get('content'), str):
                             if "【画像から読み取れた特徴（参考）】" not in last_message['content']:
                                 last_message['content'] += "\n\n" + label_text
                        # content がリストの場合 (GPT-4o/Vision用) - テキスト要素に追記
                        elif isinstance(last_message.get('content'), list):
                             text_found = False
                             for item in last_message['content']:
                                 if item.get("type") == "text":
                                     if "【画像から読み取れた特徴（参考）】" not in item.get("text",""):
                                         item["text"] = item.get("text","") + "\n\n" + label_text
                                     text_found = True
                                     break
                             if not text_found: # テキスト要素がない場合(画像のみの場合など)は新規追加
                                 last_message['content'].append({"type": "text", "text": label_text})
                        else: # 想定外の形式
                             print(f"Warning: Last message content is of unexpected type: {type(last_message.get('content'))}")
                             # 文字列として追記を試みる
                             try:
                                 current_content_str = json.dumps(last_message.get('content'))
                             except TypeError:
                                 current_content_str = str(last_message.get('content', ''))
                             last_message['content'] = current_content_str + "\n\n" + label_text

                except Exception as vision_e:
                    st.warning(f"Vision APIでの画像処理中にエラーが発生しました: {vision_e}")
                    print(f"Error during Vision API processing: {vision_e}")

            # --- 1回目のOpenAI API呼び出し ---
            print("--- Calling OpenAI API (1st time) ---")
            print(f"Messages sent (1st call):\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
            )
            response_message = response.choices[0].message
            print("--- OpenAI Response (1st time) ---")
            print(response_message)

            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                st.warning("⚠️ AIの応答が長すぎて途中で終了しました。プロンプトの指示を簡潔にするか、文字数制限を緩めてみてください。")
                print("Warning: OpenAI response finished due to length.")
            elif finish_reason != "stop" and finish_reason != "tool_calls":
                 print(f"Warning: Unexpected finish reason: {finish_reason}")

            tool_calls = response_message.tool_calls
            function_results_list = []

            if tool_calls:
                messages.append(response_message.model_dump())
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions.get(function_name)
                    if function_to_call:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"--- Calling function: {function_name} with args: {function_args} ---")
                            if function_name == 'search_google_places' and 'location_bias' not in function_args:
                                try:
                                    import streamlit as st_local
                                    if st_local.session_state.get('determined_destination_for_prompt'):
                                        coords = get_coordinates(st_local.session_state.determined_destination_for_prompt)
                                        if coords:
                                            function_args['location_bias'] = coords
                                            print(f"Added location_bias: {coords}")
                                        else:
                                            print(f"Could not get coordinates for {st_local.session_state.determined_destination_for_prompt}, proceeding without location_bias.")
                                except (ImportError, AttributeError):
                                    print("Streamlit session_state not available, skipping location_bias based on session_state.")
                            function_response_str = function_to_call(**function_args)
                            print(f"--- Function response ({function_name}) ---")
                            print(function_response_str)
                            function_results_list.append(function_response_str)
                            messages.append({
                                "tool_call_id": tool_call.id, "role": "tool",
                                "name": function_name, "content": function_response_str,
                            })
                        except json.JSONDecodeError as json_err:
                             print(f"Error decoding JSON arguments for {function_name}: {tool_call.function.arguments}. Error: {json_err}")
                             error_content_str = json.dumps({"error": f"Argument decoding error: {json_err}"}, ensure_ascii=False)
                             function_results_list.append(error_content_str)
                             messages.append({ "tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_content_str })
                        except Exception as e:
                            print(f"Error executing function {function_name} or processing its response: {e}")
                            print(traceback.format_exc())
                            error_content_str = json.dumps({"error": f"Function execution error: {str(e)}"}, ensure_ascii=False)
                            function_results_list.append(error_content_str)
                            messages.append({ "tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_content_str })
                    else:
                        print(f"Error: Function '{function_name}' not found.")
                        error_content_str = json.dumps({"error": f"Function '{function_name}' not found."}, ensure_ascii=False)
                        function_results_list.append(error_content_str)
                        messages.append({ "tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_content_str })

                print("--- Sending tool results back to OpenAI (2nd time) ---")
                print(f"Messages sent (2nd call):\n{json.dumps(messages, indent=2, ensure_ascii=False)}")
                second_response = client.chat.completions.create(model="gpt-4o", messages=messages)
                final_content = second_response.choices[0].message.content
                print("--- OpenAI Response (2nd time) ---")
                print(final_content)

                finish_reason_2 = second_response.choices[0].finish_reason
                if finish_reason_2 == "length":
                    st.warning("⚠️ AIの応答が長すぎて途中で終了しました。プロンプトの指示を簡潔にするか、文字数制限を緩めてみてください。")
                    print("Warning: OpenAI response (2nd call) finished due to length.")
                elif finish_reason_2 != "stop":
                     print(f"Warning: Unexpected finish reason (2nd call): {finish_reason_2}")

                # <<< 戻り値修正: JSON文字列のリストをそのままJSON配列文字列にする >>>
                # 各要素が有効なJSON文字列か確認
                valid_json_results = []
                for res_str in function_results_list:
                    is_valid_json = False
                    try:
                        json.loads(res_str)
                        is_valid_json = True
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in final result: {res_str}")
                        # エラー情報もJSON形式なので、そのまま追加することも可能
                        # valid_json_results.append(res_str) # エラーJSONも追加する場合
                        pass # 不正なJSONは含めない場合
                    if is_valid_json:
                         valid_json_results.append(res_str)

                # 有効なJSON文字列のリストをJSON配列文字列に変換
                final_places_data_str = json.dumps(valid_json_results, ensure_ascii=False) if valid_json_results else None
                return final_content, final_places_data_str

            else:
                print("--- No tool call requested by OpenAI ---")
                final_content = response_message.content
                return final_content, None

        except openai.APIError as e:
            st.error(f"OpenAI APIエラーが発生しました: HTTP Status={e.status_code}, Message={e.message}")
            print(f"OpenAI API Error: Status={e.status_code}, Type={e.type}, Message={e.message}")
            if e.response and hasattr(e.response, 'text'): print(f"API Response Body: {e.response.text}")
            return f"申し訳ありません、AIとの通信中にAPIエラーが発生しました。詳細: {e.message}", None
        except Exception as e:
            st.error(f"AIとの通信または関数実行中に予期せぬエラーが発生しました: {e}")
            st.error(traceback.format_exc())
            print(traceback.format_exc())
            return "申し訳ありません、処理中に予期せぬエラーが発生しました。", None

    # --- 6. Streamlitの画面構成 ---
    if "all_prefectures" not in st.session_state:
        st.session_state.all_prefectures = ["北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]
    # --- セッションステート初期化 ---
    keys_to_initialize = [
        ("show_planner_select", False), ("planner_selected", False), ("planner", None),
        ("messages", []), ("itinerary_generated", False), ("generated_shiori_content", None),
        ("final_places_data", None), ("basic_info_submitted", False),
        ("preferences_submitted", False), ("preferences", {}), ("selected_itinerary_id", None),
        ("preferences_for_prompt", {}), ("determined_destination", None),
        ("determined_destination_for_prompt", None), ("messages_for_prompt", []),
        ("shiori_name_input", ""), ("selected_itinerary_id_selector", None),
        ("main_menu", "新しい旅を計画する") # メニューのデフォルト値
    ]
    for key, default_value in keys_to_initialize:
        if key not in st.session_state:
            st.session_state[key] = default_value

# ここから上未修整（りーえー）
    # --- メインコンテンツ ---

    # --- 7. 新しい旅を計画する ---
    if menu_choice == "新しい旅を計画する":

    # 初期状態を設定
        if "started_planning" not in st.session_state:
            st.session_state.started_planning = False
        if "planner_selected" not in st.session_state:
            st.session_state.planner_selected = False
        if "show_planner_select" not in st.session_state:
            st.session_state.show_planner_select = False
        if "nickname" not in st.session_state:
            st.session_state.nickname = ""
    
    # まだ始めてないとき →「プランニングを始める」ボタンのみ表示
    if not st.session_state.started_planning:
        st.markdown('<div class="title-center">さあ、あなただけの旅をはじめよう。</div>', unsafe_allow_html=True)
        if st.button("プランニングを始める"):
            st.session_state.started_planning = True
            st.rerun()
    
    # ニックネーム入力＋「プランナーを選ぶ」ボタンを表示
    elif st.session_state.started_planning and not st.session_state.show_planner_select:
        st.subheader("あなたのニックネームを入力してください")
        st.session_state.nickname = st.text_input("ニックネーム", key="nickname_input")
        if st.button("プランナーを選ぶ"):
            if st.session_state.nickname.strip() == "":
                st.error("ニックネームを入力してください")
            else:
                st.session_state.show_planner_select = True
                st.rerun()
    # プランナー選択画面を表示する場合
    elif st.session_state.show_planner_select and not st.session_state.planner_selected:
        st.subheader("あなたにぴったりのプランナーを選んでください")
        planner_options = {
                "ベテラン": {"name": "ベテラン", "prompt_persona": "経験豊富なプロの旅行プランナーとして、端的かつ的確に", "caption": "テイスト：端的でシンプル。安心のプロ感。"},
                "姉さん": {"name": "姉さん", "prompt_persona": "地元に詳しい世話好きな姉さんとして、親しみやすい方言（例：関西弁や博多弁など、行き先に合わせて）を交えつつ元気に", "caption": "テイスト：その土地の方言＋親しみやすさ満点。"},
                "ギャル": {"name": "ギャル", "prompt_persona": "最新トレンドに詳しい旅好きギャルとして、絵文字（💖✨）や若者言葉を多用し、テンション高めに", "caption": "テイスト：テンション高め、語尾にハート。"},
                "王子": {"name": "王子", "prompt_persona": "あなたの旅をエスコートする王子様として、優雅で少しキザな言葉遣いで情熱的に", "caption": "テイスト：ちょっとナルシストだけど優しくリード。"}
            }
        col1, col2 = st.columns(2)
        with col1:
                for key in ["ベテラン", "姉さん"]:
                    st.markdown('<div class="planner-button">', unsafe_allow_html=True)
                    button_label = f"シゴデキの{key}プランナー" if key == "ベテラン" else f"地元に詳しいおせっかい{key}"
                    if st.button(button_label, key=f"planner_{key}"):
                        st.session_state.planner = planner_options[key]
                        st.session_state.planner_selected = True
                        st.session_state.step = 1
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption(planner_options[key]["caption"])
        with col2:
                 for key in ["ギャル", "王子"]:
                    st.markdown('<div class="planner-button">', unsafe_allow_html=True)
                    button_label = f"旅好きインスタグラマー（{key}）" if key == "ギャル" else f"甘い言葉をささやく{key}様"
                    if st.button(button_label, key=f"planner_{key}"):
                        st.session_state.planner = planner_options[key]
                        st.session_state.planner_selected = True
                        st.session_state.step = 1
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption(planner_options[key]["caption"])

# 898,909に「st.session_state.step = 1」追記。以下1049まで未修整（りーえー）
 
        # プランナー選択済みの場合、フォームまたは結果を表示
    elif st.session_state.planner_selected:

            # しおりが生成済みの場合
            if st.session_state.itinerary_generated and st.session_state.generated_shiori_content:
                st.header(f"旅のしおり （担当: {st.session_state.planner['name']}）")
                st.markdown(st.session_state.generated_shiori_content)
                st.markdown("---")

                # --- デバッグ情報表示 (修正版) ---
                with st.expander("▼ Function Call で取得した場所データ (デバッグ用)", expanded=False):
                    places_data_json_array_str = st.session_state.final_places_data
                    if places_data_json_array_str:
                        try:
                            # JSON配列文字列をPythonリスト(各要素はJSON文字列のはず)にパース
                            places_results_list = json.loads(places_data_json_array_str)
                            tool_call_titles = ["① 昼食候補", "② 夕食候補", "③ 宿泊候補", "④ 観光地候補"]

                            if isinstance(places_results_list, list):
                                # 各Tool呼び出しの結果を表示
                                for i, result_data in enumerate(places_results_list):
                                    title = tool_call_titles[i] if i < len(tool_call_titles) else f"Tool Call {i+1} 結果"
                                    st.subheader(title)
                                    places_data = None
                                    try:
                                        # 型チェックとパース
                                        if isinstance(result_data, str):
                                            print(f"Attempting to parse result {i} (string): {result_data[:100]}...")
                                            places_data = json.loads(result_data)
                                        elif isinstance(result_data, (list, dict)):
                                            print(f"Result {i} is already an object (type: {type(result_data)}).")
                                            places_data = result_data
                                        else:
                                            st.warning(f"予期しないデータ形式です (Type: {type(result_data)}):")
                                            st.text(str(result_data))
                                            continue

                                        # パース後のデータを処理
                                        if places_data is not None:
                                            if isinstance(places_data, list):
                                                if places_data:
                                                    try:
                                                        df = pd.DataFrame(places_data)
                                                        # マップリンク列を追加
                                                        if 'place_id' in df.columns and 'name' in df.columns:
                                                            # <<< Google Maps検索URL形式に変更 >>>
                                                            df['マップリンク'] = df.apply(
                                                                lambda row: f"[{row['name']}](https://www.google.com/maps/place/?q=place_id:{row['place_id']})", axis=1) # 修正：PlaceIDベースのURL
                                                            # 表示列を設定 (place_idは不要、マップリンク列を追加)
                                                            display_columns = ["name", "rating", "address", "マップリンク"]
                                                        else:
                                                            st.warning("place_idが見つからないため、マップリンクを生成できません。")
                                                            df['マップリンク'] = df['name'] # リンクなしの場合は名前のみ
                                                            display_columns = ["name", "rating", "address"]
                                                        # 不要な元列を削除する場合 (任意)
                                                        # if 'place_id' in df.columns: df = df.drop(columns=['place_id'])
                                                        # if 'types' in df.columns: df = df.drop(columns=['types'])
                                                        # if 'price_level' in df.columns: df = df.drop(columns=['price_level']) # 価格帯も不要な場合

                                                        # 存在する列のみを選択して表示
                                                        df_display = df[[col for col in display_columns if col in df.columns]]

                                                        # st.dataframeで表示 (Markdownリンクが解釈されるか試す)
                                                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                                                        # 代替: st.markdownでHTMLテーブルとして表示 (リンクが確実に機能)
                                                        # html_table = df_display.to_html(escape=False, index=False)
                                                        # st.markdown(html_table, unsafe_allow_html=True)
                                                    except Exception as df_e:
                                                        st.error(f"データフレーム変換/表示中にエラー: {df_e}")
                                                        st.json(places_data)
                                                else:
                                                    st.info("場所データが空です。")
                                            elif isinstance(places_data, dict):
                                                if "error" in places_data: st.error(f"エラー: {places_data['error']}")
                                                elif "message" in places_data: st.info(places_data['message'])
                                                else: st.json(places_data)
                                            else:
                                                 st.warning(f"パース後のデータ形式がリストでも辞書でもありません: {type(places_data)}")
                                                 st.text(str(places_data))
                                    except json.JSONDecodeError as json_e:
                                        st.error(f"以下のデータのJSONデコードに失敗しました: {json_e}")
                                        st.text(str(result_data))
                                    except Exception as e:
                                         st.error(f"場所データ「{title}」の表示中に予期しないエラーが発生しました: {e}")
                                         st.text(str(result_data))
                            else:
                                 st.warning("場所データの形式が予期しない形式です（リストではありません）。")
                                 st.text(places_data_json_array_str)
                        except json.JSONDecodeError:
                            st.error("場所データ全体のJSONデコードに失敗しました。")
                            st.text(places_data_json_array_str)
                        except Exception as e:
                             st.error(f"場所データの処理中にエラーが発生しました: {e}")
                             st.text(places_data_json_array_str)
                    else:
                        st.info("取得した場所データはありません。")
                # <<< デバッグ情報表示ここまで >>>

                st.markdown("---")

                # しおり保存フォーム
                with st.form("save_shiori_form"):
                    shiori_name = st.text_input("しおりの名前（保存する場合）", key="shiori_name_input", value=f"{st.session_state.get('dest', '旅行')}のしおり")
                    save_button = st.form_submit_button("このしおりを保存する")
                    if save_button:
                        if shiori_name:
                            preferences_to_save = st.session_state.get('preferences_for_prompt', {})
                            if not preferences_to_save:
                                 st.warning("保存する設定情報が見つかりません。")
                            else:
                                saved_id = save_itinerary_to_firestore(
                                    user_id, shiori_name, preferences_to_save,
                                    st.session_state.generated_shiori_content,
                                    st.session_state.final_places_data
                                )
                                if saved_id: st.success(f"しおり「{shiori_name}」を保存しました！")
                                else: st.error("しおりの保存に失敗しました。")
                        else:
                            st.warning("保存するしおりの名前を入力してください。")

                # やり直しボタン
                if st.button("条件を変えてやり直す"):
                    keys_to_clear_on_rerun = [
                        "itinerary_generated", "generated_shiori_content", "final_places_data",
                        "preferences_for_prompt", "determined_destination", "determined_destination_for_prompt",
                        "messages_for_prompt", "shiori_name_input", "basic_info_submitted",
                        "preferences_submitted", "preferences", "dest",
                        "purp", "comp", "days", "budg", "pref_nature", "pref_culture", "pref_art", "pref_welness",
                        "pref_food_local", "pref_food_style_ms", "pref_accom_type", "pref_word_ms",
                        "mbti_input", "uploaded_image_files", "q0_answer", "q1_answer", "q2_answer"
                    ]
                    for key in keys_to_clear_on_rerun:
                         if key in st.session_state: del st.session_state[key]
                    st.rerun()
# ここまで未修整（りーえー）

            # しおり未生成の場合、フォーム表示
            else:
                # 基本情報フォーム
                if 'step' not in st.session_state:
                    st.session_state.step = 1
                if st.session_state.step == 1:
                    st.subheader("1. 旅の基本情報を入力 (1/4)")
                    with st.form("basic_info_form"):
                        # 都道府県＋未定の選択肢
                        prefectures = [
                            "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
                            "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
                            "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
                            "岐阜県", "静岡県", "愛知県", "三重県",
                            "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
                            "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県",
                            "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県",
                            "沖縄県", "まだ決まっていない"
                        ]
                        st.session_state.destination = st.selectbox(
                            "目的地が決まっていたら教えてください",
                            options=prefectures,
                            index=prefectures.index(st.session_state.get("destination", "まだ決まっていない"))
                        )
                        st.session_state.comp = st.selectbox("同行者", ["一人旅", "夫婦・カップル", "友人", "家族"], index=["一人旅", "夫婦・カップル", "友人", "家族"].index(st.session_state.get('comp', '一人旅')))
                        st.session_state.days = st.number_input("旅行日数", min_value=1, max_value=30, step=1, value=st.session_state.get('days', 2))
                        st.session_state.budg = st.select_slider("予算感", options=["気にしない", "安め", "普通", "高め"], value=st.session_state.get('budg', "普通"))
                        submitted_basic = st.form_submit_button("基本情報を確定")
                        if submitted_basic:
                            st.session_state.step = 2
                            st.rerun()
                # 好み入力フォーム (基本情報入力済みの場合)
                elif st.session_state.step == 2:
                    st.info(f"基本情報を受け付けました: {st.session_state.comp}旅行 ({st.session_state.days}日間)")
                    st.subheader("2. どんな旅にしたいですか？（2/4）")
                    with st.form("destination_questions_form"):
                        current_candidates = set()
                        prefecture_questions = [
                            { "key": "q0_sea_mountain", "q": "Q1: 海と山、どっち派？", "options": ["海", "山", "どちらでも"], "mapping": { "海": ["茨城県", "千葉県", "神奈川県", "静岡県", "愛知県", "三重県", "徳島県", "香川県", "高知県", "福岡県", "佐賀県", "沖縄県", "和歌山県", "兵庫県", "岡山県", "広島県", "山口県", "愛媛県", "大分県", "宮崎県", "鹿児島県", "長崎県", "熊本県", "福井県", "石川県", "富山県", "新潟県", "東京都", "宮城県", "岩手県", "青森県", "北海道"], "山": ["山形県", "栃木県", "群馬県", "山梨県", "長野県", "岐阜県", "滋賀県", "奈良県", "埼玉県", "福島県", "秋田県"], "どちらでも": st.session_state.all_prefectures } },
                            { "key": "q1_style", "q": "Q2: 旅のスタイルは？", "options": ["アクティブに観光", "ゆったり過ごす"], "mapping": { "アクティブに観光": ["北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "長野県", "岐阜県", "静岡県", "愛知県", "三重県", "大阪府", "兵庫県", "広島県", "福岡県", "熊本県", "沖縄県"], "ゆったり過ごす": ["山梨県", "滋賀県", "京都府", "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "佐賀県", "長崎県", "大分県", "宮崎県", "鹿児島県", "沖縄県", "北海道", "青森県", "秋田県", "岩手県", "山形県", "福島県", "群馬県", "栃木県", "長野県", "岐阜県", "石川県", "富山県", "三重県", "和歌山県"] } },
                            { "key": "q2_atmosphere", "q": "Q3: どんな雰囲気を感じたい？", "options": ["和の雰囲気", "モダン・都会的", "特にこだわらない"], "mapping": { "和の雰囲気": ["青森県", "岩手県", "秋田県", "山形県", "福島県", "栃木県", "群馬県", "新潟県", "富山県", "石川県", "福井県", "岐阜県", "三重県", "滋賀県", "京都府", "奈良県", "和歌山県", "鳥取県", "島根県", "山口県", "徳島県", "愛媛県", "佐賀県", "長崎県", "熊本県", "大分県", "鹿児島県", "岡山県", "広島県", "香川県", "高知県"], "モダン・都会的": ["北海道", "宮城県", "埼玉県", "千葉県", "東京都", "神奈川県", "静岡県", "愛知県", "京都府", "大阪府", "兵庫県", "広島県", "福岡県"], "特にこだわらない": st.session_state.all_prefectures } }
                        ]
                        for i, q_data in enumerate(prefecture_questions):
                            options_with_prompt = ["選択してください"] + q_data["options"]
                            default_answer = st.session_state.get(f"q{i}_answer", "選択してください")
                            try: default_index = options_with_prompt.index(default_answer)
                            except ValueError: default_index = 0
                            st.radio(q_data["q"], options=options_with_prompt, index=default_index, key=f"q{i}_answer", horizontal=True)

                        submitted_dest = st.form_submit_button("次へ")
                        if submitted_dest:
                            unanswered = [
                                f"Q{i+1}" for i in range(len(prefecture_questions))
                                if st.session_state.get(f"q{i}_answer", "選択してください") == "選択してください"
                            ]
                            if unanswered:
                                st.warning(f"{', '.join(unanswered)} が未回答です。すべての質問に答えてください。")
                            else:
                                st.session_state.step = 3
                                st.rerun()
                elif st.session_state.step == 3:
                    st.subheader("3. あなたの好みを教えてください(3/4)")
                    with st.form("preferences_form"):
                        cols_slider = st.columns(4)
                        with cols_slider[0]: st.session_state.pref_nature = st.slider("🌲自然", 1, 5, st.session_state.get('pref_nature', 3))
                        with cols_slider[1]: st.session_state.pref_culture = st.slider("🏯歴史文化", 1, 5, st.session_state.get('pref_culture', 3))
                        with cols_slider[2]: st.session_state.pref_art = st.slider("🎨アート", 1, 5, st.session_state.get('pref_art', 3))
                        with cols_slider[3]: st.session_state.pref_welness = st.slider("♨️ウェルネス", 1, 5, st.session_state.get('pref_welness', 3))
                        cols_food = st.columns(2)
                        with cols_food[0]: st.session_state.pref_food_local = st.radio("🍽️食事場所スタイル", ["地元の人気店", "隠れ家的なお店", "シェフのこだわりのお店", "オーガニック・ヴィーガン対応のお店"], index=["地元の人気店", "隠れ家的なお店", "こだわらない"].index(st.session_state.get('pref_food_local', '地元の人気店')))
                        with cols_food[1]:
                            pref_food_style_options = ["和食", "洋食", "居酒屋", "カフェ", "スイーツ", "郷土料理", "エスニック", "ラーメン", "寿司", "中華", "イタリアン"]
                            st.session_state.pref_food_style = st.multiselect("🍲好きな料理・ジャンル", pref_food_style_options, default=st.session_state.get('pref_food_style', []), key="pref_food_style_ms")
                        st.session_state.pref_accom_type = st.radio("🏨宿タイプ", ["ホテル", "旅館", "民宿・ゲストハウス", "こだわらない"], index=["ホテル", "旅館", "民宿・ゲストハウス", "こだわらない"].index(st.session_state.get('pref_accom_type', 'ホテル')), horizontal=True)
                        pref_word_options = ["隠れた発見", "カラフル", "静かで落ち着いた", "冒険", "定番", "温泉", "寺社仏閣", "食べ歩き","ショッピング","日本酒","ワイン", "おこもり","子供と楽しむ", "ローカル体験", "アウトドア","フォトジェニック", "パワースポット", "なにもしない"]
                        st.session_state.pref_word = st.multiselect("✨気になるワード (複数選択可)", pref_word_options, default=st.session_state.get('pref_word', []), key="pref_word_ms")
                        submitted_pref_qs = st.form_submit_button("この内容で次へ")
                        if submitted_pref_qs:
                        # 回答チェック処理
                            if (
                            st.session_state.pref_food_style == "選択してください" or
                            st.session_state.pref_accom_type == "選択してください" or
                            st.session_state.pref_word == "選択してください"
                            ):
                                st.warning("すべての質問に回答してください。")
                            else:
                                st.session_state.step = 4
                                st.rerun()
                elif st.session_state.step == 4:
                    st.subheader("4. あなたのことを教えてください (4/4)")
                    with st.form("final_personal_info"):
                        st.markdown("**🧠あなたのMBTIは？（任意）**")
                        st.session_state.mbti = st.text_input("例 ENFP", value=st.session_state.get("mbti", ""), key="mbti_input", help="性格タイプに合わせて提案が変わるかも？")
                        st.markdown("**🖼️ 画像からインスピレーションを得る (任意)**")
                        uploaded_image_files = st.file_uploader("画像を3枚までアップロード", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="uploaded_image_files")
                        if uploaded_image_files and len(uploaded_image_files) > 3:
                            st.warning("画像は3枚まで。最初の3枚を使用します。")
                        st.markdown("**その他、なにかプランナーに伝えたいことはありますか？(任意)**")
                        st.text_area("例：誕生日なので思いっきりラグジュアリーにしたい！")
                        submitted_personal = st.form_submit_button("好みを確定して旅のしおりを生成✨")

 # 以下未修整（りーえー）
                        current_candidates = set(st.session_state.all_prefectures)
                        # ...(行き先決定ロジック - 変更なし)...
                        for i, q_data in enumerate(prefecture_questions):
                            answer = st.session_state.get(f"q{i}_answer")
                            if answer != "選択してください" and answer is not None:
                                mapped_prefs = set(q_data["mapping"].get(answer, st.session_state.all_prefectures))
                                current_candidates.intersection_update(mapped_prefs)

                        if not current_candidates:
                            st.warning("すべての条件に合う都道府県が見つかりませんでした。条件を少し変えてみてください。")
                            st.stop()
                        determined_destination_internal = random.choice(list(current_candidates))
                        st.session_state.determined_destination_for_prompt = determined_destination_internal
                        st.session_state.dest = determined_destination_internal

                        preferences = {
                            "nature": st.session_state.pref_nature, "culture": st.session_state.pref_culture,
                            "art": st.session_state.pref_art, "welness": st.session_state.pref_welness,
                            "food_local": st.session_state.pref_food_local, "food_style": st.session_state.pref_food_style,
                            "accom_type": st.session_state.pref_accom_type, "word": st.session_state.pref_word,
                            "mbti": st.session_state.mbti }
                        q_answers_for_pref = {q_data["key"]: st.session_state.get(f"q{i}_answer") for i, q_data in enumerate(prefecture_questions)}
                        preferences.update(q_answers_for_pref)
                        st.session_state.preferences_for_prompt = preferences
                        print(f"Preferences for prompt:\n{json.dumps(preferences, indent=2, ensure_ascii=False)}")

                        if not st.session_state.planner:
                            st.error("プランナーが選択されていません。ページをリロードしてやり直してください。")
                            st.stop()
                        navigator_persona = st.session_state.planner.get("prompt_persona", "プロの旅行プランナーとして")

                        determined_destination_for_prompt = st.session_state.determined_destination_for_prompt
                        preferences_for_prompt = st.session_state.preferences_for_prompt
                        days_for_prompt = st.session_state.days
                        purp_for_prompt = st.session_state.purp
                        comp_for_prompt = st.session_state.comp
                        budg_for_prompt = st.session_state.budg
                        food_style_list = preferences_for_prompt.get('food_style', [])
                        food_style_example = food_style_list[0] if food_style_list else "食事"
                        word_list = preferences_for_prompt.get('word', [])
                        first_word_example = word_list[0] if word_list else '観光'

                        # <<< プロンプト修正 >>>
                        prompt = f"""
あなたは旅のプランナー「Okosy」です。ユーザーの入力情報をもとに、SNS映えや定番から少し離れた、ユーザー自身の感性に寄り添うような、パーソナルな旅のしおりを作成してください。
**ユーザーに最高の旅体験をデザインすることを最優先としてください。**
**【重要】ユーザーは具体的で最新の場所情報を求めています。そのため、以下の指示に従って必ず `search_google_places` ツールを使用してください。**

【基本情報】
- 行き先: {determined_destination_for_prompt}
- 目的・気分: {purp_for_prompt}
- 同行者: {comp_for_prompt}
- 旅行日数: {days_for_prompt}日
- 予算感: {budg_for_prompt}

【ユーザーの好み】
{json.dumps(preferences_for_prompt, ensure_ascii=False, indent=2)}
★★★ 上記の好み（特に「自然」「歴史文化」「アート」「ウェルネス」の度合い、「気になるワード」、「MBTI（もしあれば）」）や、ユーザーがアップロードした好みの画像（もしあれば、画像ラベルとして後述）も考慮して、雰囲気や場所選びの参考にしてください。★★★

【出力指示】
1.  **構成:** 冒頭に、{st.session_state.planner['name']}として、なぜこの目的地({determined_destination_for_prompt})を選んだのか、どんな旅になりそうか、全体の総括を **{navigator_persona}** 言葉で語ってください。その後、{days_for_prompt}日間の旅程を、各日ごとに「午前」「午後」「夜」のセクションに分けて提案してください。時間的な流れが自然になるようにプランを組んでください。

2.  **内容:**
    * なぜその場所や過ごし方がユーザーの目的・気分・好みに合っているか、**{navigator_persona}言葉**で理由や提案コメントを添えてください。「気になるワード」の要素を意識的にプランに盛り込んでください。MBTIタイプも性格傾向を考慮するヒントにしてください。画像から読み取れた特徴も踏まえてください。
    * 隠れ家/定番のバランスはユーザーの好みに合わせてください。
    * 食事や宿泊の好みも反映してください。
    * **【説明の詳細度】** 各場所や体験について、情景が目に浮かぶような、**{navigator_persona}として感情豊かに、魅力的で詳細な説明**を心がけてください。単なるリストアップではなく、そこで感じられるであろう雰囲気や感情、おすすめのポイントなどを描写してください。ユーザーの好みを反映した説明をお願いします。（文字数の目安は設けませんが、十分な情報量を提供してください）

3.  **【場所検索の実行 - 必須】** 以下の4種類の場所について、それぞれ **必ず `search_google_places` ツールを呼び出して** 最新の情報を取得してください。取得した情報は行程提案に **必ず** 反映させる必要があります。
    * **① 昼食:** `place_type`を 'restaurant' または 'cafe' として、ユーザーの好みに合う昼食場所を検索してください。（クエリ例: "{determined_destination_for_prompt} ランチ {preferences_for_prompt.get('word', ['おしゃれ'])[0]}"）**ツール呼び出しを実行してください。**
    * **② 夕食:** `place_type`を 'restaurant' として、ユーザーの好みに合う夕食場所を検索してください。（クエリ例: "{determined_destination_for_prompt} ディナー {food_style_example} 人気"）**ツール呼び出しを実行してください。**
    * **③ 宿泊:** `place_type`を 'lodging' として、ユーザーの宿泊タイプや好みに合う宿泊施設を検索してください。（クエリ例: "{determined_destination_for_prompt} {preferences_for_prompt.get('accom_type','宿')} {preferences_for_prompt.get('word', ['温泉', '静か'])[0]}"）**ツール呼び出しを実行してください。**（宿泊タイプが「こだわらない」でも検索は実行すること）
    * **④ 観光地:** `place_type`を 'tourist_attraction', 'museum', 'park', 'art_gallery' 等からユーザーの好みに合うものを選択し、関連する観光スポットを検索してください。（クエリ例: "{determined_destination_for_prompt} {first_word_example} スポット"）**ツール呼び出しを実行してください。**

4.  **【検索結果の利用と表示】**
    * `search_google_places` ツールで得られた場所（レストラン、カフェ、宿、観光地など）を提案に含める際は、その場所名にGoogle Mapsへのリンクを **Markdown形式** で付与してください。**リンクのURLは `https://www.google.com/maps/place/?q=place_id:<PLACE_ID>` の形式**とし、`<PLACE_ID>` はツールの結果に含まれる `place_id` を使用してください。例: `[レストラン名](https://www.google.com/maps/place/?q=place_id:ChIJN1t_tDeuEmsRUsoyG83frY4)`
    * **【重要】** 場所名は**Markdownリンクの中にのみ**含めてください。リンクの前後で場所名を繰り返さないでください。
    * デバック表示で出てくるお店に関しても、同じように場所名に対してリンクが着くようにしてください(そレができればマップコードは出力不要です)
    * **各日の夜のパートには、ステップ③のツール検索結果から**、**必ず**最適な宿泊施設を1つ選び、その名前と上記形式のGoogle Mapsリンクを記載してください。もし検索結果がない場合や検索しなかった場合でも、一般的な宿泊エリアやタイプの提案をしてください。
    * 初日は必ず午前から始め、その際にホテルは出さないでください。また最終日は夜の情報を出力せずに午後で帰るようにしてください。
    * ツール検索でエラーが出たり、場所が見つからなかったりした場合は、無理に場所名を記載せず、その旨を行程中に記載してください。（例：「残念ながら条件に合う隠れ家カフェは見つかりませんでしたが、このエリアには素敵なカフェがたくさんありますよ。」）

5.  **形式:** 全体を読みやすい**Markdown形式**で出力してください。各日の区切り（例: `--- 1日目 ---`）、午前/午後/夜のセクション（例: `**午前:**`）などを明確にしてください。

{st.session_state.planner['name']}として、ユーザーに最高の旅体験をデザインしてください。
"""
                        st.session_state.messages_for_prompt = [{"role": "user", "content": prompt}]

                        
                        final_response, places_api_results_json_array_str = run_conversation_with_function_calling(
                                st.session_state.messages_for_prompt,
                                st.session_state.get("uploaded_image_files", [])
                            )

                        if final_response and "申し訳ありません" not in final_response:
                            st.session_state.itinerary_generated = True
                            st.session_state.generated_shiori_content = final_response
                            st.session_state.final_places_data = places_api_results_json_array_str
                            st.success("旅のしおりが完成しました！")
                            st.rerun()
                        else:
                            st.error("しおりの生成中にエラーが発生しました。")
                            print(f"AI Response Error or Empty: {final_response}")
                            st.session_state.itinerary_generated = False

    # --- 8. 過去の旅のしおりを見る ---
    elif menu_choice == "過去の旅のしおりを見る":
        # (過去のしおり表示部分は変更なし、デバッグ表示の修正は上記で対応済み)
        st.header("過去の旅のしおり")
        if not user_id: st.error("ユーザー情報が取得できません。"); st.stop()
        itineraries = load_itineraries_from_firestore(user_id)
        if not itineraries: st.info("まだ保存されているしおりはありません。")
        else:
            st.write(f"{len(itineraries)}件のしおりが見つかりました。")
            itinerary_options = {itin['id']: f"{itin.get('name', '名称未設定')} ({itin.get('creation_date', datetime.datetime.now(datetime.timezone.utc)).strftime('%Y-%m-%d %H:%M') if itin.get('creation_date') else '日付不明'})" for itin in itineraries}
            selected_id = st.selectbox("表示または編集/削除したいしおりを選んでください", options=list(itinerary_options.keys()), format_func=lambda x: itinerary_options[x], index=None, key="selected_itinerary_id_selector")
            st.session_state.selected_itinerary_id = selected_id
            if st.session_state.selected_itinerary_id:
                selected_itinerary = next((item for item in itineraries if item["id"] == st.session_state.selected_itinerary_id), None)
                if selected_itinerary:
                    st.subheader(f"しおり: {selected_itinerary.get('name', '名称未設定')}")
                    creation_date_utc = selected_itinerary.get('creation_date')
                    # ...(日付表示、削除ボタン、しおり内容表示は変更なし)...
                    st.markdown(selected_itinerary.get("generated_content", "コンテンツがありません。"))

                    # --- デバッグ情報表示 (過去しおり用、修正版) ---
                    st.markdown("---")
                    with st.expander("▼ 保存された場所データ (デバッグ用)"):
                        places_data_json_array_str_past = selected_itinerary.get("places_data") # 変数名変更
                        if places_data_json_array_str_past:
                            try:
                                places_results_list_past = json.loads(places_data_json_array_str_past) # 変数名変更
                                tool_call_titles_past = ["① 昼食候補", "② 夕食候補", "③ 宿泊候補", "④ 観光地候補"] # 変数名変更

                                if isinstance(places_results_list_past, list):
                                    for i, result_data_past in enumerate(places_results_list_past): # 変数名変更
                                        title_past = tool_call_titles_past[i] if i < len(tool_call_titles_past) else f"Tool Call {i+1} 結果"
                                        st.subheader(title_past)
                                        places_data_past = None # 変数名変更
                                        try:
                                            if isinstance(result_data_past, str):
                                                print(f"Attempting to parse result {i} (string): {result_data_past[:100]}...")
                                                places_data_past = json.loads(result_data_past)
                                            elif isinstance(result_data_past, (list, dict)):
                                                print(f"Result {i} is already an object (type: {type(result_data_past)}).")
                                                places_data_past = result_data_past
                                            else:
                                                st.warning(f"予期しないデータ形式です (Type: {type(result_data_past)}):")
                                                st.text(str(result_data_past))
                                                continue

                                            if places_data_past is not None:
                                                if isinstance(places_data_past, list):
                                                    if places_data_past:
                                                        try:
                                                            df_past = pd.DataFrame(places_data_past) # 変数名変更
                                                            if 'place_id' in df_past.columns and 'name' in df_past.columns:
                                                                df_past['マップリンク'] = df_past.apply(lambda row: f"[{row['name']}](https://www.google.com/maps/place/?q=place_id:{row['place_id']})", axis=1)
                                                                display_columns_past = ["name", "rating", "address", "マップリンク"]
                                                            else:
                                                                st.warning("place_idが見つからないため、マップリンクを生成できません。")
                                                                df_past['マップリンク'] = df_past['name']
                                                                display_columns_past = ["name", "rating", "address"]
                                                            df_display_past = df_past[[col for col in display_columns_past if col in df_past.columns]] # 変数名変更
                                                            st.dataframe(df_display_past, use_container_width=True, hide_index=True)
                                                        except Exception as df_e:
                                                            st.error(f"データフレーム変換/表示中にエラー: {df_e}")
                                                            st.json(places_data_past)
                                                    else: st.info("場所データが空です。")
                                                elif isinstance(places_data_past, dict):
                                                    if "error" in places_data_past: st.error(f"エラー: {places_data_past['error']}")
                                                    elif "message" in places_data_past: st.info(places_data_past['message'])
                                                    else: st.json(places_data_past)
                                                else:
                                                     st.warning(f"パース後のデータ形式がリストでも辞書でもありません: {type(places_data_past)}")
                                                     st.text(str(places_data_past))
                                        except json.JSONDecodeError as json_e:
                                            st.error(f"以下のデータのJSONデコードに失敗しました: {json_e}")
                                            st.text(str(result_data_past))
                                        except Exception as e:
                                             st.error(f"場所データ「{title_past}」の表示中に予期しないエラーが発生しました: {e}")
                                             st.text(str(result_data_past))
                                else:
                                     st.warning("場所データの形式が予期しない形式です（リストではありません）。")
                                     st.text(places_data_json_array_str_past)
                            except json.JSONDecodeError:
                                st.error("場所データ全体のJSONデコードに失敗しました。")
                                st.text(places_data_json_array_str_past)
                            except Exception as e:
                                 st.error(f"場所データの処理中にエラーが発生しました: {e}")
                                 st.text(places_data_json_array_str_past)
                        else:
                            st.info("保存された場所データはありません。")
                    # <<< デバッグ情報表示ここまで >>>

                    st.markdown("---")
                    # ...(思い出投稿フォーム、思い出一覧表示、しおり削除ボタンは変更なし)...
                    # --- 思い出投稿フォーム ---
                    st.subheader("✈️ 旅の思い出を追加")
                    with st.form(f"memory_form_{selected_itinerary['id']}", clear_on_submit=True): # clear_on_submit追加
                        memory_caption = st.text_area("思い出キャプション", key=f"memory_caption_{selected_itinerary['id']}")
                        memory_photo = st.file_uploader("思い出の写真 (任意)", type=["jpg", "jpeg", "png"], key=f"memory_photo_{selected_itinerary['id']}")
                        submit_memory = st.form_submit_button("思い出を投稿")

                        if submit_memory:
                            if memory_caption or memory_photo:
                                photo_b64 = None
                                if memory_photo:
                                    try:
                                        img_bytes = memory_photo.getvalue()
                                        photo_b64 = base64.b64encode(img_bytes).decode('utf-8')
                                    except Exception as img_e:
                                        st.warning(f"写真の処理中にエラーが発生しました: {img_e}")

                                saved_mem_id = save_memory_to_firestore(
                                    user_id, selected_itinerary['id'], memory_caption, photo_b64
                                )
                                if saved_mem_id:
                                    st.success("思い出を投稿しました！")
                                    st.rerun() # 再実行して思い出リストを更新
                                else:
                                    st.error("思い出の投稿に失敗しました。")
                            else:
                                st.warning("キャプションまたは写真を入力してください。")

                    # --- 思い出一覧表示 ---
                    st.subheader("📖 思い出アルバム")
                    memories = load_memories_from_firestore(user_id, selected_itinerary['id'])
                    if not memories:
                        st.info("このしおりにはまだ思い出が投稿されていません。")
                    else:
                        cols = st.columns(3)
                        col_index = 0
                        for memory in memories:
                            with cols[col_index % 3]:
                                st.markdown(f"**{memory.get('caption', '(キャプションなし)')}**")
                                memory_creation_date_utc = memory.get('creation_date')
                                if memory_creation_date_utc and isinstance(memory_creation_date_utc, datetime.datetime):
                                    memory_creation_date_local = memory_creation_date_utc.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
                                    st.caption(f"{memory_creation_date_local.strftime('%Y-%m-%d %H:%M')}")

                                photo_img = memory.get('photo_image')
                                if photo_img:
                                    st.image(photo_img, use_column_width=True)

                                if st.button("削除", key=f"delete_memory_{memory['id']}", help="この思い出を削除します"):
                                    if delete_memory_from_firestore(user_id, selected_itinerary['id'], memory['id']):
                                        st.success("思い出を削除しました。")
                                        st.rerun()
                                    else:
                                        st.error("思い出の削除に失敗しました。")
                                st.markdown("---")
                            col_index += 1

                    st.markdown("---")
                    # しおり削除ボタン
                    st.error("このしおりを削除する")
                    if st.button("削除を実行", key=f"delete_itinerary_{selected_itinerary['id']}", type="secondary", help="このしおりと関連する全ての思い出が削除されます。この操作は元に戻せません。"):
                        if delete_itinerary_from_firestore(user_id, selected_itinerary['id']):
                            st.success(f"しおり「{selected_itinerary.get('name', '名称未設定')}」を削除しました。")
                            st.session_state.selected_itinerary_id = None
                            st.rerun()
                        else:
                            st.error("しおりの削除に失敗しました。")
                else:
                     st.warning("選択されたしおりが見つかりませんでした。")

