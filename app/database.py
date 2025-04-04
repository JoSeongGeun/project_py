import sqlite3
import pandas as pd

DB_NAME = "wedding_halls.db"

def create_database():
    """SQLite 데이터베이스 및 테이블 생성"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS wedding_halls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        review TEXT,
        rental_fee INTEGER,
        food_price INTEGER,
        min_capacity INTEGER,
        max_capacity INTEGER,
        parking INTEGER
    )
    """)
    conn.commit()
    conn.close()
    print("✅ 데이터베이스 및 테이블이 생성되었습니다.")

def insert_data_from_csv(csv_file="wedding_halls.csv"):
    """CSV 데이터를 데이터베이스에 삽입"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_csv(csv_file)
    
    df.to_sql("wedding_halls", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("✅ 데이터가 데이터베이스에 저장되었습니다.")

def fetch_all_halls():
    """모든 예식장 데이터 가져오기"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM wedding_halls", conn)
    conn.close()
    return df

# 실행 (최초 1회)
if __name__ == "__main__":
    create_database()
    insert_data_from_csv()
