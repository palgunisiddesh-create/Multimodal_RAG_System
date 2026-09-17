from database import create_database, check_login, get_users

create_database()

print("Database created successfully!")

user = check_login("admin", "admin123")

if user:
    print("Login successful!")
    print("Username:", user["username"])
    print("Role:", user["role"])
    print("Status:", user["status"])
else:
    print("Login failed!")

print("\nUsers:")

users = get_users()

for user in users:
    print(user)