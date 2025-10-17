"""
Authentication Module

This module provides a simplified form of user authentication and account management functionality.

"""

import box
import yaml
from typing import Optional
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against a bcrypt hashed password.
    
    Args:
        plain_password (str): The plain text password to verify.
        hashed_password (str): The bcrypt hashed password from storage.
        
    Returns:
        bool: True if the password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)

def load_users() -> dict:
    """
    Load user configuration from YAML file.
    
    Reads the users configuration from '../config/users.yml' and returns
    the users dictionary.
    
    Returns:
        dict: Dictionary mapping usernames to user details (hashed_password, role).
    """
    with open('config/users.yml', 'r', encoding='utf8') as ymlfile:
        users_cfg = box.Box(yaml.safe_load(ymlfile))
    return users_cfg.users

def authenticate(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user and return their role if credentials are valid.
    Args:
        username (str): The username to authenticate.
        password (str): The plain text password to verify.
        
    Returns:
        Optional[str]: User's role (e.g., 'user', 'admin') if authentication succeeds,
                      None if authentication fails.
    """
    users = load_users()
    user = users.get(username)
    if user and verify_password(password, user.hashed_password):
        return user.role

    return None

def create_account() -> None:
    """
    CLI interface for creating a new user account.
    Prompts the user to enter a username and password, validates that the username
    is unique, hashes the password, and saves the new user to the YAML configuration.
    
    Returns:
        None
    """
    print("="*60)
    print("\nSIGN UP FOR NEW USER\n")
    print("="*60)

    users = load_users()

    username = input("Choose a USERNAME: ")
    if username in users:
        print("\nUSERNAME ALREADY EXISTS. TRY AGAIN.\n")
        return create_account()

    password = input("Choose a PASSWORD: ")
    hashed_password = generate_hash(password)
    users[username] = {
        'hashed_password': hashed_password,
        'role': 'user'
    }
    save_user_hashes(users)
    print("\nUSER REGISTERED SUCCESSFULLY!\n")

def assign_role(username: str, role: str) -> None:
    """
    Assign or update a role for an existing user.
    
    Changes the role of a user in the YAML configuration. Prevents modification
    of admin user roles for security reasons.
    
    Args:
        username (str): The username to update.
        role (str): The new role to assign (e.g., 'user', 'admin').
        
    Returns:
        None
        
    Notes:
        - Admin user roles cannot be changed
        - Prints status messages to stdout
    """
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

def access() -> Optional[str]:
    """
    CLI interface for user authentication.
    
    Prompts the user to enter credentials and validates them. Recursively calls
    itself if authentication fails.
    
    Returns:
        Optional[str]: User's role if authentication succeeds.
    """
    print("="*60)
    print("\nAUTHENTICATION REQUIRED\n")
    print("="*60)

    role = authenticate(input("USERNAME: "), input("PASSWORD: "))
    if not role:
        print("\nAUTHENTICATION FAILED\n")
        access()

    return role


def generate_hash(password: str) -> str:
    """
    Generate a bcrypt hash for a plain text password.
    
    Args:
        password (str): The plain text password to hash.
        
    Returns:
        str: The bcrypt hashed password.
    """
    return pwd_context.hash(password)


def save_user_hashes(users: dict, file_path: str = '../config/users.yml') -> None:
    """
    Save user credentials to YAML file.
    
    Converts the users dictionary to the appropriate YAML format and writes it to
    the specified file path.
    
    Args:
        users (dict): Dictionary mapping usernames to user details (password/hashed_password, role).
        file_path (str): Path to save the YAML file. Defaults to '../config/users.yml'.
    """
    users_cfg = {'users': {}}
    for username, details in users.items():
        hashed_password = details.get('hashed_password')
        if not hashed_password and 'password' in details:
            hashed_password = generate_hash(details['password'])
        if not hashed_password:
            raise ValueError(f"Missing password data for user '{username}'.")
        users_cfg['users'][username] = {
            'hashed_password': hashed_password,
            'role': details['role']
        }

    with open(file_path, 'w', encoding='utf8') as ymlfile:
        yaml.dump(users_cfg, ymlfile)


def create_account_streamlit(username: str, password: str) -> bool:
    """
    Streamlit interface for creating a new user account.
    
    Creates a new user account with the provided username and password.
    Raises an exception if the username already exists.
    
    Args:
        username (str): The username for the new account.
        password (str): The password for the new account.
        
    Returns:
        bool: True if account creation succeeds.
    """
    users = load_users()

    if username in users:
        raise Exception("Username already exists.")

    users[username] = {
        'hashed_password': generate_hash(password),
        'role': 'user'
    }
    save_user_hashes(users)
    return True


def access_streamlit(username: str, password: str) -> str:
    """
    Streamlit interface for user authentication.
    
    Validates the provided credentials against the stored user configuration.
    
    Args:
        username (str): The username to authenticate.
        password (str): The plain text password to verify.
        
    Returns:
        str: User's role if authentication succeeds.
    """
    users = load_users()
    user = users.get(username)

    if not user:
        raise Exception("Invalid username or password.")

    if verify_password(password, user.hashed_password):
        return user.role

    raise Exception("Invalid username or password.")