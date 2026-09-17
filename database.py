import sqlite3
import hashlib

DATABASE_NAME = "users.db"


def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_database():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            uploaded_by TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    )

    if cursor.fetchone() is None:
        cursor.execute("""
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, ?)
        """, (
            "admin",
            hash_password("admin123"),
            "admin",
            "active"
        ))

    connection.commit()
    connection.close()


def check_login(username, password):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT username, role, status
        FROM users
        WHERE username = ? AND password = ?
    """, (username, hash_password(password)))

    user = cursor.fetchone()
    connection.close()

    if user and user[2] == "active":
        return {
            "username": user[0],
            "role": user[1],
            "status": user[2]
        }

    return None


def add_user(username, password, role="user"):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO users (username, password, role, status)
            VALUES (?, ?, ?, ?)
        """, (username, hash_password(password), role, "active"))
        connection.commit()
        result = True
    except sqlite3.IntegrityError:
        result = False

    connection.close()
    return result


def get_users():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("SELECT id, username, role, status FROM users ORDER BY id")
    users = cursor.fetchall()
    connection.close()
    return users


def delete_user(user_id):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    connection.commit()
    connection.close()


def change_status(user_id, status):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
    connection.commit()
    connection.close()


def add_document(filename, uploaded_by):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    try:
        cursor.execute("""
            INSERT INTO documents (filename, uploaded_by)
            VALUES (?, ?)
        """, (filename, uploaded_by))
        connection.commit()
        result = True
    except sqlite3.IntegrityError:
        result = False

    connection.close()
    return result


def get_documents():
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()
    cursor.execute("""
        SELECT id, filename, uploaded_by, uploaded_at
        FROM documents
        ORDER BY id DESC
    """)
    documents = cursor.fetchall()
    connection.close()
    return documents


def delete_document(document_id):
    connection = sqlite3.connect(DATABASE_NAME)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT filename FROM documents WHERE id = ?",
        (document_id,)
    )
    row = cursor.fetchone()

    if row:
        filename = row[0]
        cursor.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,)
        )
        connection.commit()
    else:
        filename = None

    connection.close()
    return filename
