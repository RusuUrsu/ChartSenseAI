import box
import yaml
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def load_users():
    with open('../config/users.yml', 'r', encoding='utf8') as ymlfile:
        users_cfg = box.Box(yaml.safe_load(ymlfile))
    return users_cfg.users

def authenticate(username, password):
    users = load_users()
    user = users.get(username)
    if user and verify_password(password, user.hashed_password):
        return user.role

    return None

def create_account():
    print("="*60)
    print("\nSIGN UP FOR NEW USER\n")
    print("="*60)

    users = load_users()

    username = input("Choose a USERNAME: ")
    if username in users:
        print("\nUSERNAME ALREADY EXISTS. TRY AGAIN.\n")
        return create_account()

    password = input("Choose a PASSWORD: ")
    user = (username, {'password': password, 'role': 'user'})
    users[username] = user[1]
    save_user_hashes(users)
    print("\nUSER REGISTERED SUCCESSFULLY!\n")

def assign_role(username, role):
    users = load_users()
    if username in users:
        if users[username]['role'] != 'admin':
            users[username]['role'] = role
            save_user_hashes(users)
        else:
            print("\nCANNOT CHANGE ROLE OF ADMIN USER.\n")
            return
        print(f"\nROLE '{role}' ASSIGNED TO USER '{username}' SUCCESSFULLY!\n")
    else:
        print("\nUSERNAME NOT FOUND. CANNOT ASSIGN ROLE.\n")

def access():
    print("="*60)
    print("\nAUTHENTICATION REQUIRED\n")
    print("="*60)

    role = authenticate(input("USERNAME: "), input("PASSWORD: "))
    if not role:
        print("\nAUTHENTICATION FAILED\n")
        access()

    return role


def generate_hash(password):
    return pwd_context.hash(password)


def save_user_hashes(users, file_path='../config/users.yml'):
    users_cfg = {'users': {}}
    for username, details in users.items():
        users_cfg['users'][username] = {
            'hashed_password': generate_hash(details['hashed_password']),
            'role': details['role']
        }

    with open(file_path, 'w', encoding='utf8') as ymlfile:
        yaml.dump(users_cfg, ymlfile)


def create_account_streamlit(username: str, password: str):
    """
    Create a new account for Streamlit frontend.
    """
    users = load_users()

    if username in users:
        raise Exception("Username already exists.")

    # Add user with hashed password
    users[username] = {
        'password': password,
        'role': 'user'
    }
    save_user_hashes(users)
    return True


def access_streamlit(username: str, password: str):
    """
    Authenticate user for Streamlit frontend.
    Returns the user's role if authentication succeeds.
    """
    users = load_users()
    user = users.get(username)

    if not user:
        raise Exception("Invalid username or password.")

    if verify_password(password, user.hashed_password):
        return user.role

    raise Exception("Invalid username or password.")
